import numpy as np
import torch
import torch.nn as nn
import os
import sys
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_curve, auc
import csv
import matplotlib
import matplotlib.pyplot as plt
import json
from arg_config import parse_args
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def _safe_auc(y, s):
    try:
        if len(np.unique(y)) > 1:
            return roc_auc_score(y, s)
        return float("nan")
    except Exception:
        return float("nan")


def _safe_acc(y, yhat):
    try:
        if len(np.unique(y)) > 1:
            return accuracy_score(y, yhat)
        return float("nan")
    except Exception:
        return float("nan")


def evaluate_thresholds_df(df: pd.DataFrame, t12: float, t34: float):
    """
    Given a subject-level df with columns ['true_label','class_4_percentage'],
    evaluate Mean AUC / Mean ACC for the two clinical subtasks:
    - S4 vs S1-3 (p4)
    - S1 vs S2-4 (p1)
    using the existing get_probabilities() mapping.
    """
    assert {"true_label","class_4_percentage"} <= set(df.columns), \
        "[evaluate_thresholds_df] df must contain true_label and class_4_percentage."
    x = df["class_4_percentage"].values.astype(float)
    y_true = df["true_label"].values
    y_true_s4 = (y_true == 4).astype(int)
    y_true_s1 = (y_true == 1).astype(int)

    p4, p1 = get_probabilities(x, t12, t34, eps=1e-6)
    acc_s4 = _safe_acc(y_true_s4, (p4 >= 0.5).astype(int))
    acc_s1 = _safe_acc(y_true_s1, (p1 >= 0.5).astype(int))
    auc_s4 = _safe_auc(y_true_s4, p4)
    auc_s1 = _safe_auc(y_true_s1, p1)

    mean_acc = np.nanmean([acc_s4, acc_s1]) if not (np.isnan(acc_s4) and np.isnan(acc_s1)) else float("nan")
    mean_auc = np.nanmean([auc_s4, auc_s1]) if not (np.isnan(auc_s4) and np.isnan(auc_s1)) else float("nan")
    return dict(mean_auc=mean_auc, mean_acc=mean_acc,
                auc_s4=auc_s4, auc_s1=auc_s1, acc_s4=acc_s4, acc_s1=acc_s1)


def boxplot_initial_thresholds(df: pd.DataFrame):
    """
    Compute initial thresholds from boxplot-style stats.
    Prefer class-wise Q3/Q1 midpoints if labels available; otherwise fall back to global quartiles.
    Returns (t12_init, t34_init).
    """
    vals = df["class_4_percentage"].astype(float)
    present = sorted(df["true_label"].unique().tolist())
    # default global quartiles
    q25, q75 = float(vals.quantile(0.25)), float(vals.quantile(0.75))
    t12_init, t34_init = q25, q75

    # Try class-wise if enough classes
    group_stats = df.groupby("true_label")["class_4_percentage"].describe()
    # order classes by class median (50%)
    order = group_stats["50%"].sort_values().index.tolist()
    if len(order) >= 2:
        a, b = order[0], order[1]
        t12_init = 0.5 * (group_stats.loc[a, "75%"] + group_stats.loc[b, "25%"])
    if len(order) >= 4:
        c, d = order[-2], order[-1]
        t34_init = 0.5 * (group_stats.loc[c, "75%"] + group_stats.loc[d, "25%"])
    elif len(order) >= 3:
        # if only three groups, split between middle and top
        c, d = order[1], order[2]
        t34_init = 0.5 * (group_stats.loc[c, "75%"] + group_stats.loc[d, "25%"])

    # clamp to [0,1]
    t12_init = float(np.clip(t12_init, 0.0, 1.0))
    t34_init = float(np.clip(t34_init, 0.0, 1.0))
    return t12_init, t34_init


def tune_thresholds_locally(df: pd.DataFrame, base_t12: float, base_t34: float,
                            step: float = 0.01, delta: float = 0.05,
                            min_gap: float = 0.03, prefer: str = "auc"):
    """
    Locally tune thresholds around boxplot-initial values on the validation set.
    - Search t12 in [base_t12-delta, base_t12+delta]
    - Search t34 in [base_t34-delta, base_t34+delta]
    - Enforce t34 - t12 >= min_gap
    - Optimize mean AUC by default; fall back to mean ACC if AUC is NaN.
    """
    lo12, hi12 = max(0.0, base_t12 - delta), min(1.0, base_t12 + delta)
    lo34, hi34 = max(0.0, base_t34 - delta), min(1.0, base_t34 + delta)
    grid12 = np.arange(lo12, hi12 + 1e-12, step)
    grid34 = np.arange(lo34, hi34 + 1e-12, step)

    best = {"score": -np.inf, "t12": base_t12, "t34": base_t34, "mean_auc": float("nan"), "mean_acc": float("nan")}
    for t12 in grid12:
        for t34 in grid34:
            if (t34 - t12) < min_gap:
                continue
            m = evaluate_thresholds_df(df, t12, t34)
            score = m["mean_auc"] if (prefer == "auc" and not np.isnan(m["mean_auc"])) else m["mean_acc"]
            if np.isnan(score):
                continue
            if score > best["score"]:
                best.update({"score": float(score), "t12": float(t12), "t34": float(t34),
                             "mean_auc": float(m["mean_auc"]) if not np.isnan(m["mean_auc"]) else float("nan"),
                             "mean_acc": float(m["mean_acc"]) if not np.isnan(m["mean_acc"]) else float("nan")})
    return best


config = parse_args()

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


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, fold_idx):
    """ 训练 ResNet 模型 """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('train model to device', flush=True)

    # 初始化最佳验证指标
    train_losses, val_losses = [], []
    train_accuracy, val_accuracy = [], []

    best_epoch = -1
    best_model_path = os.path.join(config.fold_path, f'best_model_fold_{fold_idx}.pth')
    model_path = os.path.join(config.fold_path, f'model.pth')

    subject_acc = []
    best_val_acc = 0.0  # 这里存放的是 MeanAUC（可退化为 MeanACC）
    val_losses, val_acc = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, label = [], []
        val_loss, val_acc, val_total, val_f1 = 0.0, 0, 0, 0
        batch_count = 0

        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.float().to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_count += 1
            if batch_count % 2000 == 0:
                batch_loss = loss.item()
                _, preds = torch.max(outputs, 1)
                batch_correct = (preds == labels).float().mean().item()
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_count}], Loss: {batch_loss:.4f}, Accuracy: {batch_correct:.4f}", flush=True)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            label.extend(labels.cpu().numpy())

        train_acc = correct / total
        train_balanced_acc = balanced_accuracy_score(label, all_preds)
        train_f1_score = f1_score(label, all_preds, average='weighted')
        train_accuracy.append(train_acc)
        train_losses.append(total_loss)

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {train_acc:.4f}, Balanced Accuracy: {train_balanced_acc:.4f}, F1-score: {train_f1_score:.4f}]", flush=True)

        # —— 验证阶段（subject-level 聚合 + 阈值本地微调）—— #
        subject_patch_percentages = inference_model(model, val_loader)

        # 映射真实标签到 {1,2,3,4} 并构建 df
        subject_classifications = {}
        for subject_id, data in subject_patch_percentages.items():
            true_label = data['true_label']
            if true_label == 0:
                true_label = 1
            elif true_label == 1:
                true_label = 4
            class_4_percentage = data['class_4_percentage']
            subject_classifications[subject_id] = {
                'true_label': true_label,
                'class_4_percentage': class_4_percentage
            }

        df = pd.DataFrame([
            {
                "subject_id": sid,
                "true_label": data["true_label"],
                "class_4_percentage": np.mean(data["class_4_percentage"])
            }
            for sid, data in subject_classifications.items()
        ])

        # —— 阈值：当折内验证集缺少 S2/S3 时，跳过本地微调，仅用箱线图初值 —— #
        labels_present = set(df["true_label"].unique().tolist())
        t12_init, t34_init = boxplot_initial_thresholds(df)
        print(f"[BoxPlot init] t12_init={t12_init:.4f}, t34_init={t34_init:.4f}", flush=True)

        if not ({2, 3} & labels_present):
            # 验证集只有 {1,4}（或不含 2/3），不做本地阈值微调，避免对两类分布过拟合
            print("[Warn] Fold-val contains only {1,4} (no 2/3). Skip local threshold tuning; use boxplot initials.", flush=True)
            tuned = {
                "t12": float(t12_init),
                "t34": float(t34_init),
                "mean_auc": float("nan"),
                "mean_acc": float("nan"),
            }
        else:
            # 2) 本地微调（val 集含 2/3 时才进行）
            tuned = tune_thresholds_locally(
                df, base_t12=t12_init, base_t34=t34_init,
                step=0.005, delta=0.05, min_gap=0.03, prefer="auc"
            )
            print(
                f"[Tuned] t12={tuned['t12']:.4f}, t34={tuned['t34']:.4f} | "
                f"MeanAUC={tuned['mean_auc']}, MeanACC={tuned['mean_acc']}",
                flush=True,
            )

        threshold_1_vs_rest = tuned["t12"]
        threshold_4_vs_rest = tuned["t34"]

        # 用分段线性概率计算 AUC/ACC
        True_labels = df["true_label"].values
        predicted_probabilities = df["class_4_percentage"].values
        p4, p1 = get_probabilities(predicted_probabilities, threshold_1_vs_rest, threshold_4_vs_rest)
        y_true_s4 = (True_labels == 4).astype(int)
        y_true_s1 = (True_labels == 1).astype(int)

        y_pred_s4 = (p4 >= 0.5).astype(int)
        y_pred_s1 = (p1 >= 0.5).astype(int)
        acc_s4 = accuracy_score(y_true_s4, y_pred_s4)
        acc_s1 = accuracy_score(y_true_s1, y_pred_s1)
        try:
            auc_s4 = roc_auc_score(y_true_s4, p4)
        except ValueError:
            auc_s4 = float("nan")
        try:
            auc_s1 = roc_auc_score(y_true_s1, p1)
        except ValueError:
            auc_s1 = float("nan")

        mean_acc = 0.5 * (acc_s4 + acc_s1)
        mean_auc = (auc_s4 + auc_s1) / 2.0 if (not np.isnan(auc_s4) and not np.isnan(auc_s1)) else float("nan")

        print(f"[VAL@epoch {epoch+1}] "
              f"S4vsS123 ACC={acc_s4:.3f}, AUC={auc_s4:.3f} | "
              f"S1vsS234 ACC={acc_s1:.3f}, AUC={auc_s1:.3f} | "
              f"MeanACC={mean_acc:.3f}, MeanAUC={mean_auc:.3f}", flush=True)

        # —— 以 MeanAUC 选最佳（tie 用 MeanACC）并保存阈值 —— #
        is_better = False
        if np.isnan(mean_auc):
            is_better = (mean_acc > best_val_acc)
        else:
            if mean_auc > best_val_acc:
                is_better = True
            elif abs(mean_auc - best_val_acc) < 1e-6 and mean_acc > getattr(config, "best_val_mean_acc", -1):
                is_better = True

        if is_better:
            best_val_acc = mean_auc if not np.isnan(mean_auc) else mean_acc
            config.best_val_mean_acc = mean_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            thr_json = os.path.join(config.fold_path, f'best_thresholds_fold_{fold_idx}.json')
            with open(thr_json, "w") as f:
                json.dump({"t12": float(threshold_1_vs_rest), "t34": float(threshold_4_vs_rest)}, f, indent=2)
            print(f"[✔] Best model saved (epoch {best_epoch}). MeanAUC={best_val_acc:.4f}, thresholds saved to {thr_json}", flush=True)

    return best_model_path, best_val_acc  # best_val_acc stores MeanAUC when available, else MeanACC


def validate_model(model, val_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds, all_labels, all_subjects, all_probs = [], [], [], []
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    final_subject_probs = 0.0
    val_f1 = 0

    with torch.no_grad():
        for imgs, labels, subjects_id in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if isinstance(subjects_id, tuple):
                subjects_id = list(subjects_id)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_subjects.extend(subjects_id)

    val_acc = val_correct / val_total
    val_balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='binary')
    val_loss /= len(val_loader)
    return val_acc, val_balanced_acc, val_loss, val_f1


def inference_model(model, inference_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('inference start')
    model.to(device)
    model.eval()
    all_preds, all_subjects, all_labels = [], [], []

    with torch.no_grad():
        for batch in inference_data:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 4:
                    imgs, labels, subjects_id, _positions = batch
                elif len(batch) == 3:
                    imgs, labels, subjects_id = batch
                else:
                    raise ValueError(f"Unexpected number of items from dataloader: {len(batch)}")
            else:
                raise ValueError("Dataloader must return a tuple/list")
            imgs, labels = imgs.to(device), labels.to(device)
            if isinstance(subjects_id, tuple):
                subjects_id = list(subjects_id)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_subjects.extend(subjects_id)
            all_labels.extend(labels.cpu().numpy())

    subject_votes = defaultdict(list)
    subject_labels = defaultdict(list)
    for pred, subject, labels in zip(all_preds, all_subjects, all_labels):
        subject_votes[subject].append(pred)
        subject_labels[subject] = labels

    subject_patch_percentages = {}
    for subject, votes in subject_votes.items():
        total_patche = len(votes)
        class_1_percentage = votes.count(1) / total_patche
        # NOTE: 在当前二分类设置下，patch 预测类别 1 被视作 "Stage-4-like"。
        # 因此 class_1_percentage 代表 subject 的 "class_4_percentage_from_patches" 的估计值。
        subject_patch_percentages[subject] = {
            "class_4_percentage": class_1_percentage,  # ≙ class_4_percentage_from_patches
            "true_label": subject_labels[subject],
        }
    return subject_patch_percentages


def val_model(model, val_loader, criterion):
    subject_ids = set()
    for imgs, ids, positions in val_loader:
        subject_ids.update(ids)
    print("val_model统计到的subject数：", len(subject_ids))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('inference start')
    model.to(device)
    model.eval()

    all_preds, all_subjects, all_positions = [], [], []

    with torch.no_grad():
        for imgs, subjects_id, positions in val_loader:
            imgs = imgs.to(device)
            if isinstance(subjects_id, tuple):
                subjects_id = list(subjects_id)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_subjects.extend(subjects_id)
            all_positions.extend(positions)

    subject_votes = defaultdict(list)
    subject_positions = defaultdict(list)
    for pred, subject, position in zip(all_preds, all_subjects, all_positions):
        subject_votes[subject].append(pred)
        subject_positions[subject].append(position)

    subject_patch_percentages = {}
    for subject, votes in subject_votes.items():
        total_patche = len(votes)
        class_1_percentage = votes.count(1) / total_patche
        subject_patch_percentages[subject] = {
            "class_4_percentage": class_1_percentage,
        }
    return subject_patch_percentages