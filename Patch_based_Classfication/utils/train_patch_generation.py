import torch 
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import csv
import glob
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
from sklearn.model_selection import StratifiedKFold
from nilearn import image as nli
import pandas as pd
import gc
import argparse
from scipy.ndimage import zoom
import nibabel as nib
import ants
import SimpleITK as sitk
from arg_config import parse_args
from utils.pre_processing import classes_management, get_class, split_data, expand_data_for_balance, create_kfolds, create_test_data


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



class MRIPatch_dataset():
    def __init__(self, mri_path, mask_path, patch_size, cover_rate, max_patches_per_slice=100):
        """
        initial MRI Patch dataset
        :param mri_path: MRI path
        :param mask_path: Mask path
        :param patch_size: Patch size
        :param cover_rate: threshold for patch coverage
        :param max_patches_per_slice: patch number per slice
        """
        self.mri_path = mri_path
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.cover_rate = cover_rate
        self.max_patches_per_slice = max_patches_per_slice

    def extract_patches(self):
        patches = []
        patch_positions = []

        extracted_rate = 0.1  

        mask_img = nli.load_img(self.mask_path)
        mask_data = np.array(mask_img.dataobj, dtype=np.float32)
        standard_shape = (mask_data.shape[0], mask_data.shape[1], mask_data.shape[2])

        mri_data_list = []
        for path in self.mri_path:
            if path == '0':
                mri_data = np.ones(standard_shape, dtype=np.float32)
                mri_data_list.append(mri_data)
             
            else:
                mri_data = np.array(nli.load_img(path).dataobj, dtype=np.float32)
                mri_data_list.append(mri_data)
            


        mri_data_list_safe = []
        for data in mri_data_list:
            mean = float(np.mean(data))
            std = float(np.std(data))
            if std <1e-8:
                mri_data_list_safe.append(np.zeros_like(data,dtype=np.float32))
            else:
                norm = ((data - mean) / (std + 1e-8)).astype(np.float32)
                mri_data_list_safe.append(norm)
        mri_data_list = mri_data_list_safe

        num_slices = mask_data.shape[2]
        half_size = self.patch_size // 2
        

        for i in range(num_slices):
            mask_slice = mask_data[:, :, i]
            positions = np.argwhere(mask_slice == 1)[:, :2]

            slice_patches = []
            slice_positions = []

            for x, y in positions:
                if (x - half_size >= 0 and x + half_size < mask_slice.shape[0] and
                    y - half_size >= 0 and y + half_size < mask_slice.shape[1]):
                    
                    patch_area = mask_slice[x-half_size:x+half_size, y-half_size:y+half_size]
                    coverage = np.sum(patch_area) / (self.patch_size ** 2)

              
                    if coverage >= self.cover_rate:
                        patch = np.stack([m[x-half_size:x+half_size, y-half_size:y+half_size, i] for m in mri_data_list], axis=-1)
                        # print('patch.shape', patch.shape, flush=True)
                        slice_patches.append(patch)
                        slice_positions.append((i, x, y))


            if len(slice_patches) > 0:
                num_to_extract = max(1, int(len(slice_patches) * extracted_rate))  # 至少提取一个
                indices = np.random.choice(len(slice_patches), num_to_extract, replace=False)
                slice_patches = [slice_patches[j] for j in indices]
                slice_positions = [slice_positions[j] for j in indices]

            patches.extend(slice_patches)
            patch_positions.extend(slice_positions)

            # release 

            del slice_patches, slice_positions
            gc.collect()

        return patches, patch_positions


### overcome oom write in the .npy files
def save_patches_to_npy(data, labels, fold_num, data_type, patch_size, cover_rate, stage_type, patch_batch_size=50000):
    all_patches = []
    all_labels = []
    all_id = []
    all_positions = []
    chunk_index = 0
    total_patches = 0

    folder = os.path.join(args.fold_path, f"fold_{fold_num}")
    if not os.path.exists(folder):
        os.makedirs(folder)

    for idx, (item, label) in enumerate(zip(data, labels)):
        mri_path = item['mri_path']
        mask_path = item['mask_path']
        # print('mask_path', mask_path, flush=True)  
        if mask_path is None or not os.path.exists(mask_path):
            print(f"Warning: mask file missing for subject {mask_path}, skipping.", flush=True)
            print(f"mri_path: {mri_path}", flush=True)
            continue
         
        file_name = item['filename']
        print(f"Processing subject {idx + 1}/{len(data)}: {file_name} with label {label}", flush=True)
        if stage_type =="train":
            patch_dataset = MRIPatch_dataset(mri_path, mask_path, patch_size, cover_rate)
        else:
            patch_dataset = MRIPatch_dataset(mri_path, mask_path, patch_size, cover_rate)

        patch_data, patch_position = patch_dataset.extract_patches()
        
        if patch_data is None or len(patch_data) == 0:
            # print(f"Warning: No patches extracted for subject {file_name}, skipping.", flush=True)
            continue
        else:
            all_patches.extend(patch_data)

            # print(f"Total patches extracted: {len(patch_data)} for subject {file_name}", flush=True)
            all_positions.extend(patch_position)
            all_id.extend([file_name] * len(patch_position))
            all_labels.extend([label] * len(patch_position))
            total_patches += len(patch_data)

        while len(all_patches) >= patch_batch_size:
            # print(f"Saving chunk part_{chunk_index} with {patch_batch_size} patches...", flush=True)
            target_shape = (patch_size, patch_size, 3)
            filtered_patches = []
            filtered_positions = []
            filtered_ids = []
            filtered_labels = []
            for i, p in enumerate(all_patches[:patch_batch_size]):
                arr = np.array(p)
                if arr.shape == target_shape:
                    filtered_patches.append(arr)
                    filtered_positions.append(all_positions[i])
                    filtered_ids.append(all_id[i])
                    filtered_labels.append(all_labels[i])
                else:
                    print(f"Warning: patch {i} shape {arr.shape} != {target_shape}, will be skipped.")

            if len(filtered_patches) == 0:
                print(f"Error: No valid patches in chunk {chunk_index}, skipping save.")
            else:
                np.save(os.path.join(folder, f"{data_type}_data_part_{chunk_index}.npy"), np.stack(filtered_patches, axis=0))
                np.save(os.path.join(folder, f"{data_type}_labels_part_{chunk_index}.npy"), np.array(filtered_labels))
                np.save(os.path.join(folder, f"{data_type}_id_part_{chunk_index}.npy"), np.array(filtered_ids))
                np.save(os.path.join(folder, f"{data_type}_positions_part_{chunk_index}.npy"), np.array(filtered_positions))

            # 清理已保存部分
            all_patches = all_patches[patch_batch_size:]
            all_labels = all_labels[patch_batch_size:]
            all_id = all_id[patch_batch_size:]
            all_positions = all_positions[patch_batch_size:]
            chunk_index += 1
            gc.collect()


    # 写入最后剩余的内容
    if len(all_patches) > 0:
        print(f"Saving final chunk part_{chunk_index} with {len(all_patches)} patches...", flush=True)
        # np.save(os.path.join(folder, f"{data_type}_data_part_{chunk_index}.npy"), np.array(all_patches))
        np.save(os.path.join(folder, f"{data_type}_data_part_{chunk_index}.npy"),np.stack(all_patches[:patch_batch_size], axis=0))
        np.save(os.path.join(folder, f"{data_type}_labels_part_{chunk_index}.npy"), np.array(all_labels))
        np.save(os.path.join(folder, f"{data_type}_id_part_{chunk_index}.npy"), np.array(all_id))
        np.save(os.path.join(folder, f"{data_type}_positions_part_{chunk_index}.npy"), np.array(all_positions))
        chunk_index += 1
        gc.collect()

    print(f"Total patches saved: {total_patches} in {chunk_index} chunk(s).", flush=True)

    
            
if __name__ == "__main__":
    args = parse_args()

    set_seed(42)  # randome seeds

    class_number = 4
    category_dict = get_class(class_number)   

    category_counts = {key: len(value) for key, value in category_dict.items()}
    train_val_data, test_data, inference_data = split_data(category_dict)
    all_test_data = create_test_data(train_val_data, test_data, inference_data)


    # check the extention
    for category, items in train_val_data.items():
        print(f"Category {category}: {len(items)} samples")

    print("\nTest Data:")
    for category, items in test_data.items():
        print(f"Category {category}: {len(items)} samples")

    #keep balance
    expanded_train_val_data = expand_data_for_balance(train_val_data)

    for category, items in expanded_train_val_data.items():
        print(f"Expanded Category {category}: {len(items)} samples")
    kfolds = create_kfolds(expanded_train_val_data)
    patch_size = args.patch_size
    cover_rate = args.cover_rate

    for fold_num,fold_data in enumerate(kfolds):
        print(f"Processing fold {fold_num + 1}...")
        save_patches_to_npy(
            data=fold_data['train'],
            labels=fold_data['train_labels'],
            fold_num=fold_num + 1,
            data_type="train",
            patch_size=args.patch_size,
            cover_rate=args.cover_rate,
            stage_type="train"
        )

        save_patches_to_npy(
            data=fold_data['val'],
            labels=fold_data['val_labels'],
            fold_num=fold_num + 1,
            data_type="val",
            patch_size=args.patch_size, 
            cover_rate=args.cover_rate,
            stage_type="train"
        )

    print("All folds processed successfully.")


    test_items = []
    test_labels = []

    for category, items in all_test_data.items():
        test_items.extend(items)  
        test_labels.extend([category] * len(items))  

    # save test data
    save_patches_to_npy(
        data=test_items,
        labels=test_labels,
        fold_num="test", 
        data_type="test",  
        patch_size=args.patch_size, 
        cover_rate=args.cover_rate,
        stage_type="test" 
        )



