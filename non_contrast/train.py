import os, sys
import torch
from  utils.trainer import train_model, inference_model, boxplot_initial_thresholds, tune_thresholds_locally, get_probabilities
from utils.data_loader import  New_BalancedBatchSampler,BalancedSubjectBatchSampler,BalancedBatchSampler, Muliti_MRI_Dataset, Muliti_val_Dataset
from torch.utils.data import DataLoader
from utils.model_class import classification_resnet_model
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from collections import Counter
from arg_config import parse_args
import glob, json
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.metrics import roc_auc_score as _auc
import random
from utils.post_processing import get_probabilities, _eval_metrics_with_piecewise, ensemble_inference_on_4class_set, _load_fold_models

# ---- Seed helper (E) ----
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---- Evaluate with global thresholds (C) ----
def evaluate_with_global_thresholds(source_folder, df):
    """Load best_models/global_thresholds.json, compute metrics using piecewise-linear probabilities."""
    thr_path = os.path.join(source_folder, "best_models", "global_thresholds.json")
    assert os.path.exists(thr_path), f"Global thresholds not found: {thr_path}"
    with open(thr_path, "r") as f:
        g = json.load(f)
    t12, t34 = float(g["t12"]), float(g["t34"])

    # Compute metrics strictly via piecewise-linear probabilities (get_probabilities)
    metrics = _eval_metrics_with_piecewise(df, t12=t12, t34=t34)
    print("[Ensemble Evaluation]",
          f"MeanAUC={metrics['mean_auc']}",
          f"MeanACC={metrics['mean_acc']}",
          f"| AUC_S4={metrics['auc_s4']} ACC_S4={metrics['acc_s4']}",
          f"| AUC_S1={metrics['auc_s1']} ACC_S1={metrics['acc_s1']}")
    return metrics, (t12, t34)


def get_matching_data_label_pairs(fold_path, stage_type):
    data_paths = sorted(glob.glob(os.path.join(fold_path, f"{stage_type}_data_part_*.npy")))
    label_paths = sorted(glob.glob(os.path.join(fold_path, f"{stage_type}_labels_part_*.npy")))
    id_paths = sorted(glob.glob(os.path.join(fold_path, f"{stage_type}_id_part_*.npy")))

    data_dict = {re.search(r"part_(\d+)", path).group(1): path for path in data_paths if re.search(r"part_(\d+)", path)}
    label_dict = {re.search(r"part_(\d+)", path).group(1): path for path in label_paths if re.search(r"part_(\d+)", path)}
    id_dict = {re.search(r"part_(\d+)", path).group(1): path for path in id_paths if re.search(r"part_(\d+)", path)}

    matched_keys = sorted(set(data_dict.keys()) & set(label_dict.keys()) & set(id_dict.keys()), key=int)
    matched_data = [data_dict[k] for k in matched_keys]
    matched_labels = [label_dict[k] for k in matched_keys]
    matched_ids = [id_dict[k] for k in matched_keys]
    return matched_data, matched_labels, matched_ids


def get_all_data_label_pairs(fold_path):
    def extract_parts(paths, prefix):
        result = {}
        for path in paths:
            match = re.search(r"part_(\d+)", path)
            if match:
                key = f"{prefix}_{match.group(1)}"
                result[key] = path
        return result
    train_data_paths = sorted(glob.glob(os.path.join(fold_path, "train_data_part_*.npy")))
    train_label_paths = sorted(glob.glob(os.path.join(fold_path, "train_labels_part_*.npy")))
    train_id_paths = sorted(glob.glob(os.path.join(fold_path, "train_id_part_*.npy")))

    val_data_paths = sorted(glob.glob(os.path.join(fold_path, "val_data_part_*.npy")))
    val_label_paths = sorted(glob.glob(os.path.join(fold_path, "val_labels_part_*.npy")))
    val_id_paths = sorted(glob.glob(os.path.join(fold_path, "val_id_part_*.npy")))

    data_dict = {
        **extract_parts(train_data_paths, "train"),
        **extract_parts(val_data_paths, "val")
    }
    label_dict = {
        **extract_parts(train_label_paths, "train"),
        **extract_parts(val_label_paths, "val")
    }
    id_dict = {
        **extract_parts(train_id_paths, "train"),
        **extract_parts(val_id_paths, "val")
    }

    matched_keys = sorted(set(data_dict) & set(label_dict) & set(id_dict))
    matched_data = [data_dict[k] for k in matched_keys]
    matched_labels = [label_dict[k] for k in matched_keys]
    matched_ids = [id_dict[k] for k in matched_keys]
    return matched_data, matched_labels, matched_ids


def with_all_cross_validation(source_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)  # (E) 固定随机性

    best_models_dir = os.path.join(source_folder, "best_models")
    os.makedirs(best_models_dir, exist_ok=True)

    fold_results = []
    test_folder = os.path.join(source_folder, "fold_test")
    print('test_folder', test_folder, flush=True)

    test_data_paths, test_label_paths, test_id_paths = get_matching_data_label_pairs(test_folder, "test")
    print("test_data_paths:", len(test_data_paths))
    test_dataset = Muliti_val_Dataset(test_data_paths, test_label_paths, test_id_paths)
    test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    for fold_idx in range(1, 5):
        print(f"\n🚀 Training on Fold {fold_idx}/4", flush=True)
        fold_path = os.path.join(source_folder, f"fold_{fold_idx}")
        print('fold_path', fold_path, flush=True)

        train_data_paths, train_label_paths, train_id_paths = get_matching_data_label_pairs(fold_path, "train")
        val_data1_paths, val_label1_paths, val_ids1_paths = get_matching_data_label_pairs(fold_path, "val")

        val_data_paths = val_data1_paths
        val_label_paths = val_label1_paths
        val_ids_paths   = val_ids1_paths

        print("train_data_paths:", len(train_data_paths))
        print("val_data_paths:", len(val_data_paths))

        train_data = Muliti_MRI_Dataset(train_data_paths, train_label_paths, train_id_paths)
        val_data = Muliti_val_Dataset(val_data_paths, val_label_paths, val_ids_paths)

        all_labels = []
        for i in range(len(train_data)):
            _, label, _ = train_data[i]
            all_labels.append(int(label))
        label_counter = Counter(all_labels)
        print(f"Fold {fold_idx} - Train label distribution:")
        for label, count in sorted(label_counter.items()):
            print(f"  Class {label}: {count} patches")

        batch_size = config.batch_size
        sampler = New_BalancedBatchSampler(train_data, batch_size)
        train_loader = DataLoader(train_data,  batch_sampler=sampler)
        val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
        print('data loaded finished', flush=True)
        print(f"Total train batches per epoch: {len(train_loader)}")

        model = classification_resnet_model(num_modalities=3).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=config.lr_step_size, gamma = config.lr_gamma)
        num_epochs = config.num_epochs

        _old_fold_path = getattr(config, 'fold_path', None)
        config.fold_path = fold_path
        best_model_path, best_val_acc = train_model(
            model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, fold_idx
        )
        if _old_fold_path is not None:
            config.fold_path = _old_fold_path

        fold_results.append((fold_idx, best_model_path, best_val_acc))
        print(f"Best model for Fold {fold_idx} saved at {best_model_path} with MeanAUC/ACC score: {best_val_acc:.4f}", flush=True)

        expected_fold_best = os.path.join(fold_path, f"best_model_fold_{fold_idx}.pth")
        if not os.path.exists(expected_fold_best):
            try:
                state = torch.load(best_model_path, map_location=device)
                torch.save(state, expected_fold_best)
                print(f"[i] Copied best model to {expected_fold_best} for ensemble stage.")
            except Exception as e:
                print(f"[!] Could not materialize expected fold best model: {e}")

        # —— using the best module to generate t —— #
        try:
            single_model = classification_resnet_model(num_modalities=3).to(device)
            state = torch.load(best_model_path, map_location=device)
            single_model.load_state_dict(state)
            subj_dict = inference_model(single_model, test_loader)
            rows = []
            for sid, data in subj_dict.items():
                tl = data['true_label']
                if tl == 0:
                    tl_mapped = 1
                elif tl == 1:
                    tl_mapped = 4
                else:
                    tl_mapped = int(tl)
                p4 = float(data['class_4_percentage'])
                rows.append({
                    'subject_id': sid,
                    'true_label': tl_mapped,
                    'class_4_percentage': p4,
                    'class_1_percentage': 1.0 - p4,
                })
            oof_df = pd.DataFrame(rows)
            oof_csv_path = os.path.join(fold_path, f"oof_subject_fold_{fold_idx}.csv")
            oof_df.to_csv(oof_csv_path, index=False)
            print(f"[✔] Saved OOF subject predictions for fold {fold_idx} to {oof_csv_path}")

            try:
                t12_init, t34_init = boxplot_initial_thresholds(oof_df)
                tuned_local = tune_thresholds_locally(
                    oof_df, base_t12=t12_init, base_t34=t34_init,
                    step=0.005, delta=0.05, min_gap=0.03, prefer="auc"
                )
                t12_loc, t34_loc = float(tuned_local["t12"]), float(tuned_local["t34"])

                x_vals = oof_df["class_4_percentage"].values.astype(float)
                p4_vals, p1_vals = get_probabilities(x_vals, t12=t12_loc, t34=t34_loc)

                y4 = (oof_df["true_label"].values == 4).astype(int)
                y1 = (oof_df["true_label"].values == 1).astype(int)

                fpr4, tpr4, _ = roc_curve(y4, p4_vals)
                fpr1, tpr1, _ = roc_curve(y1, p1_vals)
                auc4 = roc_auc_score(y4, p4_vals) if (y4.sum() > 0 and (1-y4).sum() > 0) else float("nan")
                auc1 = roc_auc_score(y1, p1_vals) if (y1.sum() > 0 and (1-y1).sum() > 0) else float("nan")
                
                ### draw ROC curve
                plt.figure()
                plt.plot(fpr4, tpr4, label=f'S4 vs S123 (AUC={auc4:.3f})')
                plt.plot(fpr1, tpr1, label=f'S1 vs S234 (AUC={auc1:.3f})')
                plt.plot([0,1], [0,1], linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Fold {fold_idx} ROC (piecewise probs)')
                plt.legend(loc='lower right')
                roc_path = os.path.join(fold_path, f"fold_{fold_idx}_roc.png")
                plt.savefig(roc_path, bbox_inches='tight', dpi=200)
                plt.close()
                print(f"[✔] Saved ROC curve for fold {fold_idx} → {roc_path}")

                try:
                    groups = []
                    labels_order = [1, 2, 3, 4]
                    for lab in labels_order:
                        groups.append(oof_df.loc[oof_df['true_label'] == lab, 'class_4_percentage'].values)

                    # 使用白底网格风格
                    try:
                        plt.style.use('seaborn-v0_8-whitegrid')
                    except Exception:
                        plt.style.use('ggplot')

                    fig, ax = plt.subplots(figsize=(8, 6))
                    bp = ax.boxplot(
                        groups,
                        tick_labels=[str(l) for l in labels_order],  # Matplotlib 3.9+: use tick_labels
                        showmeans=False,
                        patch_artist=True,   # 允许填充色
                        widths=0.6,
                        whis=1.5
                    )

                    # 颜色与线型
                    for box in bp['boxes']:
                        box.set(facecolor='C0', alpha=0.35, edgecolor='black', linewidth=1.2)
                    for median in bp['medians']:
                        median.set(color='black', linewidth=2.0)
                    for whisker in bp['whiskers']:
                        whisker.set(color='0.4', linewidth=1.2)
                    for cap in bp['caps']:
                        cap.set(color='0.4', linewidth=1.2)
                    if 'means' in bp:
                        for mean in bp['means']:
                            mean.set(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=4)

                    ax.set_xlabel('True Label')
                    ax.set_ylabel('Class 4 Percentage')
                    ax.set_title('Class 4 Percentage Distribution by True Label')
                    ax.set_ylim(0.0, 1.0)
                    ax.grid(True, axis='y', linestyle='-', alpha=0.25)

                    box_path = os.path.join(fold_path, f"fold_{fold_idx}_boxplot.png")
                    fig.savefig(box_path, bbox_inches='tight', dpi=200)
                    plt.close(fig)
                    print(f"[✔] Saved box plot for fold {fold_idx} → {box_path}")
                except Exception as e_box:
                    print(f"[!] Failed to create box plot for fold {fold_idx}: {e_box}")

            except Exception as e_roc:
                print(f"[!] Failed to create ROC for fold {fold_idx}: {e_roc}")
        except Exception as e:
            print(f"[!] Failed to generate OOF predictions for fold {fold_idx}: {e}")

    best_fold = max(fold_results, key=lambda x: x[2])
    best_fold_idx, best_model_path, best_val_acc = best_fold
    final_model_path = os.path.join(best_models_dir, f"best_model.pth")
    torch.save(torch.load(best_model_path), final_model_path)
    print(f"\n🏆 Best model is from Fold {best_fold_idx} with MeanAUC/ACC score: {best_val_acc:.4f}")
    print(f"Best model path: {best_model_path}")

    try:
        oof_csv_paths = []
        for k in range(1, 5):
            p = os.path.join(source_folder, f"fold_{k}", f"oof_subject_fold_{k}.csv")
            if os.path.exists(p):
                oof_csv_paths.append(p)
            else:
                print(f"[!] OOF CSV missing for fold {k}: {p}")
        if len(oof_csv_paths) == 0:
            print("[!] No OOF CSVs found; skip global threshold tuning.")
        else:
            df_list = [pd.read_csv(p) for p in oof_csv_paths]
            oof_df = pd.concat(df_list, ignore_index=True)
            assert {"true_label","class_4_percentage"} <= set(oof_df.columns), \
                "OOF CSV must contain columns true_label and class_4_percentage."

            # (C/F) Sanity AUC 检查
            try:
                y4_sanity = (oof_df['true_label'].values == 4).astype(int)
                s4_sanity = oof_df['class_4_percentage'].values.astype(float)
                auc4_sanity = _auc(y4_sanity, s4_sanity)
                if auc4_sanity < 0.5:
                    print(f"[Warn] Sanity AUC(S4 vs S123)={auc4_sanity:.3f} < 0.5. Check score direction/columns.")
            except Exception as _e:
                print(f"[Warn] Sanity AUC check failed: {_e}")

            t12_init, t34_init = boxplot_initial_thresholds(oof_df)
            tuned = tune_thresholds_locally(
                oof_df, base_t12=t12_init, base_t34=t34_init,
                step=0.005, delta=0.05, min_gap=0.03, prefer="auc"
            )
            global_t12 = float(tuned["t12"])
            global_t34 = float(tuned["t34"])
            print(f"[Global Thresholds] t12={global_t12:.4f}, t34={global_t34:.4f} | MeanAUC={tuned['mean_auc']} MeanACC={tuned['mean_acc']}")

            global_thr_path = os.path.join(best_models_dir, "global_thresholds.json")
            with open(global_thr_path, "w") as f:
                json.dump({"t12": global_t12, "t34": global_t34}, f, indent=2)
            print(f"[✔] Global thresholds saved to {global_thr_path}")
    except Exception as e:
        print(f"[!] Global threshold tuning failed: {e}")

    print("\n✅ Cross-validation completed successfully!")
    return best_model_path


if __name__ == "__main__":
    config = parse_args()
    source_folders = config.fold_path
    print(f"Starting k-fold cross-validation on source folder: {source_folders}", flush=True)
    best_model_path = with_all_cross_validation(source_folders)
    # load the best model path
    best_model_path = os.path.join(source_folders, "best_models", "best_model.pth")

    # === Optional full-data retraining (disabled by default) ===
    ENABLE_FULL_RETRAIN = False
    if ENABLE_FULL_RETRAIN:
        print("[Info] Full-data retrain enabled.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = classification_resnet_model(num_modalities=3).to(device)
        model.load_state_dict(torch.load(best_model_path))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

        k_fold_path = os.path.join(source_folders, "fold_1")
        full_train_data_paths, full_train_label_paths, full_train_id_paths = get_all_data_label_pairs(k_fold_path)
        full_train_data = Muliti_MRI_Dataset(full_train_data_paths, full_train_label_paths, full_train_id_paths)
        sampler = BalancedSubjectBatchSampler(full_train_data, config.batch_size)
        full_train_loader = DataLoader(full_train_data, batch_sampler=sampler)
        num_epochs_full = config.num_epochs

        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        best_loss = float('inf')
        for epoch in range(1, num_epochs_full + 1):
            model.train()
            epoch_loss, total, correct = 0.0, 0, 0
            for batch_idx, (images, labels, ids) in enumerate(full_train_loader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            scheduler.step()
            avg_loss = epoch_loss / max(1, len(full_train_loader))
            accuracy = 100.0 * correct / max(1, total)
            print(f"[FullRetrain] Epoch [{epoch}/{num_epochs_full}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                torch.save(model.state_dict(), os.path.join(source_folders, "best_models", "full_retrain_best.pth"))
                print("[FullRetrain] ↳ improved; saved to best_models/full_retrain_best.pth")
        final_model_path = os.path.join(source_folders,"best_models","final_full_model.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"\n💾 Final model saved to: {final_model_path}")
    else:
        print("[Info] Skipping full-data retrain (ensemble-first).")

    # —— 4 fold ensemble on fold_test —— #
    df_ens = ensemble_inference_on_4class_set(source_folders, batch_size=config.batch_size)
    metrics, (t12, t34) = evaluate_with_global_thresholds(source_folders, df_ens)
    df_ens.to_csv(os.path.join(source_folders, "best_models", "ensemble_subject_oof.csv"), index=False)
    print(f"[✔] Ensemble subject-level predictions saved. Global thresholds: t12={t12:.4f}, t34={t34:.4f}")

    # —— 在 ensemble 分数上再微调一次阈值（C） —— #
    try:
        t12_i, t34_i = boxplot_initial_thresholds(df_ens)
        tuned_ens = tune_thresholds_locally(
            df_ens, base_t12=t12_i, base_t34=t34_i,
            step=0.005, delta=0.05, min_gap=0.03, prefer="auc"
        )
        metrics_tuned = _eval_metrics_with_piecewise(df_ens, t12=float(tuned_ens['t12']), t34=float(tuned_ens['t34']))
        print("[Ensemble Tuned]",
              f"MeanAUC={metrics_tuned['mean_auc']}",
              f"MeanACC={metrics_tuned['mean_acc']}",
              f"| AUC_S4={metrics_tuned['auc_s4']} ACC_S4={metrics_tuned['acc_s4']}",
              f"| AUC_S1={metrics_tuned['auc_s1']} ACC_S1={metrics_tuned['acc_s1']}",
              f"| t12={float(tuned_ens['t12']):.4f} t34={float(tuned_ens['t34']):.4f}")
    except Exception as e:
        print(f"[!] Ensemble local tuning failed: {e}")