import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import gc
from arg_config import parse_args  
from patch_generation import get_class,MRIPatch_dataset

from model_class import classification_resnet_model
import csv
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


### non-contrast的数据可以直接使用这个来加载 不会报oom
class TrainPatch_dataset(Dataset):
    def __init__(self, data, position, id):
        self.data = data
        self.position = position
        self.id = id
    

    def __getitem__(self, index):
        data = self.data[index]

        if isinstance(data, np.ndarray):
            if data.ndim == 4:
                mid = data.shape[-1] // 2
                data = data[..., mid]
            if data.ndim == 3:
                if data.shape[0] in [1, 3, 7]:  # [C, H, W]
                    pass  
                elif data.shape[-1] in [1, 3, 7]:  # [H, W, C]
                    data = np.transpose(data, (2, 0, 1))  # → [C, H, W]
                else:
                    raise ValueError(f"Unexpected 3D shape: {data.shape}")
            else:
                raise ValueError(f"Unexpected shape: {data.shape}")
        else:
            raise ValueError(f"Unexpected data type: {type(data)}")

        data = torch.tensor(data, dtype=torch.float32)
        position = self.position[index]
        id = self.id[index]
        return data, id, position

    def __len__(self):
        return len(self.data)
    



#### 使用线性的函数来进行threshold的调整
def get_probabilities(x_vals, t12, t34, eps=1e-12):
    """
    Compute monotone piecewise linear probabilities p4 and p1.

    Args:
        x_vals (float or array-like): class_4_percentage values in [0,1].
        t12 (float): threshold for Stage 1 vs Stage 2-4 (for p1).
        t34 (float): threshold for Stage 4 vs Stage 1-3 (for p4).
        eps (float): small value to avoid division by zero.

    Returns:
        p4 (np.ndarray or float): probabilities for Stage 4 vs Stage 1-3 (increasing).
        p1 (np.ndarray or float): probabilities for Stage 1 vs Stage 2-4 (decreasing).
    """
    x = np.atleast_1d(x_vals).astype(float)

    # p4 increasing function
    t34_safe = max(t34, eps)
    one_minus_t34 = max(1.0 - t34, eps)
    p4 = np.empty_like(x)
    left = (x <= t34)
    right = ~left
    p4[left] = 0.5 * (x[left] / t34_safe)
    p4[right] = 0.5 + 0.5 * ((x[right] - t34) / one_minus_t34)
    p4 = np.clip(p4, 0.0, 1.0)

    # p1 decreasing function
    t12_safe = max(t12, eps)
    one_minus_t12 = max(1.0 - t12, eps)
    p1 = np.empty_like(x)
    left = (x <= t12)
    right = ~left
    p1[left] = 1.0 - 0.5 * (x[left] / t12_safe)
    p1[right] = 0.5 - 0.5 * ((x[right] - t12) / one_minus_t12)
    p1 = np.clip(p1, 0.0, 1.0)

    if np.isscalar(x_vals):
        return float(p4[0]), float(p1[0])
    return p4, p1





if __name__ == "__main__":
    config = parse_args() 
    folder_file = config.input_dir
    mask_file = config.mask_dir
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_dict = get_class(folder_file, mask_file,config.data_type)  
    if config.data_type == "NonContrast":
        num_modalities = 3
        model_path = os.path.join(base_dir,'model','non_contrast_model.pth')
        Threshold = [0.3659 ,0.6614]
        threshold_1_2 = Threshold[0]
        threshold_3_4 = Threshold[1]
    else:
        num_modalities = 7
        model_path =os.path.join(base_dir,'model','contrast_model.pth')
        Threshold = [0.3315, 0.7221]  # 1-2, 2-3, 3-4 thresholds
        threshold_1_2 = Threshold[0]
        threshold_3_4 = Threshold[1]




    ### module load
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(model_path):
        logging.error(f'Model weights not found: {model_path}')
        raise SystemExit(1)
    model = classification_resnet_model(num_modalities).to(device)
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
    except Exception as e:
        logging.error(f'Failed to load model weights from {model_path}: {e}')
        raise SystemExit(1)
    model.eval()

    cirterion = nn.CrossEntropyLoss()


    results = []
    model.eval()

    for file_name, items in file_dict.items():
        print(f'Processing {file_name} with {len(items)} items...',flush=True)
        mri_path = [item['mri_path'] for item in items]
        mask_path = [item['mask_path'] for item in items]

        val_dataset = MRIPatch_dataset(mri_path, mask_path, config.patch_size, config.cover_rate, config.data_type)
        val_patch, val_patch_position = val_dataset.extract_patches()
        # print(f"Extracted {len(val_patch)} patches.", flush=True)
        try:
            val_dataset = MRIPatch_dataset(mri_path, mask_path, config.patch_size, config.cover_rate, config.data_type)
            val_patch, val_patch_position = val_dataset.extract_patches()
        except Exception as e:
            logging.warning(f'[{file_name}] patch extraction failed: {e}')
            continue
        if len(val_patch) == 0:
            logging.warning(f'[{file_name}] no patches extracted, skipping.')
            continue

        try:
            subject_dataset = TrainPatch_dataset(val_patch, val_patch_position, [file_name] * len(val_patch))
            bs = getattr(config, "batch_size", 16)
            total_preds = []
            try:
                subject_loader = DataLoader(subject_dataset, batch_size=bs, shuffle=False, num_workers=0)
                with torch.no_grad():
                    for imgs, _, _ in subject_loader:
                        imgs = imgs.to(device, non_blocking=True)
                        preds = torch.softmax(model(imgs), dim=1)[:, 1]
                        total_preds.append(preds.cpu())
            except torch.cuda.OutOfMemoryError as oom:
                logging.warning(f'[{file_name}] OOM at batch_size={bs}, retry with batch_size=1. Detail: {oom}')
                torch.cuda.empty_cache()
                subject_loader = DataLoader(subject_dataset, batch_size=1, shuffle=False, num_workers=0)
                with torch.no_grad():
                    for imgs, _, _ in subject_loader:
                        imgs = imgs.to(device, non_blocking=True)
                        preds = torch.softmax(model(imgs), dim=1)[:, 1]
                        total_preds.append(preds.cpu())
        except Exception as e:
            logging.warning(f'[{file_name}] inference failed: {e}')
            continue

        total_preds = torch.cat(total_preds).numpy()
        if not np.isfinite(total_preds).all():
            logging.warning(f'[{file_name}] predictions contain NaN/Inf, skipping case.')
            continue

        class4_percentage = float(np.mean(total_preds))
        # Piecewise mapping: extremes use direct mapping; middle uses fuzzy (as requested)
        p_4_vs_123_piecewise, p_1_vs_234_piecewise =get_probabilities(class4_percentage,threshold_1_2, threshold_3_4, eps=1e-12)
        results.append({
            "Case": file_name,
            "Setting": config.data_type,
            "Subtask1_prob_S4": round(p_4_vs_123_piecewise, 4),
            "Subtask2_prob_S1": round(p_1_vs_234_piecewise, 4),
        })
        torch.cuda.empty_cache()
        gc.collect()

    #### 将fuzzyset的结果保存为csv文件 
    csv_file_path = os.path.join(config.output_dir, "LiFS_pred.csv")
    try:
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        with open(csv_file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Case", "Setting",
                "Subtask1_prob_S4",
                "Subtask2_prob_S1",
            ])
            writer.writeheader()
            writer.writerows(results)
        logging.info(f'Wrote results to {csv_file_path} (cases={len(results)})')
    except Exception as e:
        logging.error(f'Failed to write CSV {csv_file_path}: {e}')
        raise
