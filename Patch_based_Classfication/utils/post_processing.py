import os, sys
from  utils.trainer import train_model, inference_model, boxplot_initial_thresholds, tune_thresholds_locally, get_probabilities
from utils.data_loader import  New_BalancedBatchSampler,BalancedSubjectBatchSampler,BalancedBatchSampler, Muliti_MRI_Dataset, Muliti_val_Dataset
from utils.model_class import classification_resnet_model
from collections import defaultdict
import numpy as np
import glob
import re
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from collections import Counter
from arg_config import parse_args
import glob
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import pandas as pd
from torch.utils.data import DataLoader
import torch


# --- Piecewise metrics helper (already used everywhere) --- back processing function 
def get_probabilities(x_vals, t12, t34, eps=1e-12):
    x = np.atleast_1d(x_vals).astype(float)
    t34_safe = max(t34, eps)
    one_minus_t34 = max(1.0 - t34, eps)
    p4 = np.empty_like(x)
    left = (x <= t34)
    right = ~left
    p4[left] = 0.5 * (x[left] / t34_safe)
    p4[right] = 0.5 + 0.5 * ((x[right] - t34) / one_minus_t34)
    p4 = np.clip(p4, 0.0, 1.0)

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


def _eval_metrics_with_piecewise(df, t12, t34):
    """Compute S4vsS123 / S1vsS234 AUC & ACC using piecewise-linear probabilities.
    Expects df with columns: true_label (1..4), class_4_percentage in [0,1].
    NOTE: Metrics are computed ONLY via get_probabilities (piecewise-linear),
    not by raw score thresholding or any legacy method.
    """
    x = df["class_4_percentage"].values.astype(float)
    p4, p1 = get_probabilities(x, t12=t12, t34=t34)

    # S4 vs S123
    y4 = (df["true_label"].values == 4).astype(int)
    auc_s4 = np.nan
    try:
        auc_s4 = roc_auc_score(y4, p4)
    except ValueError:
        auc_s4 = float("nan")
    acc_s4 = accuracy_score(y4, (p4 >= 0.5).astype(int))

    # S1 vs S234
    y1 = (df["true_label"].values == 1).astype(int)
    auc_s1 = np.nan
    try:
        auc_s1 = roc_auc_score(y1, p1)
    except ValueError:
        auc_s1 = float("nan")
    acc_s1 = accuracy_score(y1, (p1 >= 0.5).astype(int))

    mean_auc = np.nanmean([auc_s4, auc_s1])
    mean_acc = (acc_s4 + acc_s1) / 2.0
    return {
        "mean_auc": float(mean_auc),
        "mean_acc": float(mean_acc),
        "auc_s4": float(auc_s4) if auc_s4 == auc_s4 else float("nan"),
        "acc_s4": float(acc_s4),
        "auc_s1": float(auc_s1) if auc_s1 == auc_s1 else float("nan"),
        "acc_s1": float(acc_s1),
    }






def _load_fold_models(source_folder, num_modalities=3, device=None):
    """Load best_model_fold_k.pth for k=1..4 and return a list of models on device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = []
    for k in range(1, 5):
        fold_dir = os.path.join(source_folder, f"fold_{k}")
        expected = os.path.join(fold_dir, f"best_model_fold_{k}.pth")
        chosen = None
        if os.path.exists(expected):
            chosen = expected
        else:
            print(f"[!] Missing model for fold {k}: {expected}")
            # 1) Try root-level best_model_fold_{k}.pth
            root_expected = os.path.join(source_folder, f"best_model_fold_{k}.pth")
            if os.path.exists(root_expected):
                chosen = root_expected
                print(f"[i] Found root-level checkpoint for fold {k}: {os.path.basename(chosen)}")
            else:
                # 2) Discover alternative checkpoints inside fold_k
                try:
                    pth_files = sorted(glob.glob(os.path.join(fold_dir, "*.pth")))
                    if pth_files:
                        best_like = [p for p in pth_files if "best" in os.path.basename(p).lower()]
                        candidates = best_like if best_like else pth_files
                        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                        chosen = candidates[0]
                        print(f"[i] Using discovered checkpoint for fold {k}: {os.path.basename(chosen)}")
                    else:
                        print(f"[i] No .pth files found under {fold_dir}")
                except Exception as e:
                    print(f"[!] Error while scanning {fold_dir} for checkpoints: {e}")
        if chosen is not None:
            model_paths.append(chosen)

    models = []
    for mp in model_paths:
        m = classification_resnet_model(num_modalities=num_modalities).to(device)
        m.load_state_dict(torch.load(mp, map_location=device))
        m.eval()
        models.append(m)

    if len(models) == 0:
        # Fallback: try single best model in best_models
        single_best = os.path.join(source_folder, 'best_models', 'best_model.pth')
        if os.path.exists(single_best):
            print(f"[!] No per-fold models found. Falling back to single best model: {single_best}")
            m = classification_resnet_model(num_modalities=num_modalities).to(device)
            m.load_state_dict(torch.load(single_best, map_location=device))
            m.eval()
            models.append(m)
        else:
            raise FileNotFoundError("No fold models found and no best_models/best_model.pth available.")

    # 明确打印加载的模型清单（F）
    print("[Ensemble] Loaded models:")
    for path in model_paths:
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = None
        print(f"  - {path} (mtime={mtime})")

    return models



def ensemble_inference_on_4class_set(source_folder, batch_size=64):
    """
    Run 4-fold ensemble (average probs) on the extra 4-class validation set (fold_test).
    Returns a DataFrame with columns: [subject_id, true_label, class_4_percentage, class_1_percentage].
    - class_4_percentage is the mean of the ensemble positive-class probabilities per subject.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _load_fold_models(source_folder, num_modalities=3, device=device)

    # Build the 4-class validation ("test") dataloader
    test_folder = os.path.join(source_folder, "fold_test")
    test_data_paths, test_label_paths, test_id_paths = get_matching_data_label_pairs(test_folder, "test")
    dataset = Muliti_val_Dataset(test_data_paths, test_label_paths, test_id_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    from collections import defaultdict
    subj_probs = defaultdict(list)
    subj_true = {}

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 4:
                    imgs, labels, subjects_id, _positions = batch
                elif len(batch) == 3:
                    imgs, labels, subjects_id = batch
                else:
                    raise ValueError(f"Unexpected number of items from dataloader: {len(batch)}")
            else:
                raise ValueError("Dataloader must return a tuple/list")

            imgs = imgs.to(device)
            labels = labels.cpu().numpy()

            probs_stack = []
            for m in models:
                logits = m(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1]  # positive class prob (Stage-4)
                probs_stack.append(probs.unsqueeze(0))
            avg_probs = torch.mean(torch.cat(probs_stack, dim=0), dim=0).cpu().numpy()

            if isinstance(subjects_id, tuple):
                subjects_id = list(subjects_id)
            for sid, y, p in zip(subjects_id, labels, avg_probs):
                if y == 0:
                    y_mapped = 1
                elif y == 1:
                    y_mapped = 4
                else:
                    y_mapped = int(y)
                subj_true[sid] = y_mapped
                subj_probs[sid].append(float(p))

    rows = []
    for sid, plist in subj_probs.items():
        _p4 = float(np.mean(plist))
        rows.append({
            "subject_id": sid,
            "true_label": subj_true[sid],
            "class_4_percentage": _p4,
            "class_1_percentage": 1.0 - _p4,
        })
    df = pd.DataFrame(rows)
    return df
