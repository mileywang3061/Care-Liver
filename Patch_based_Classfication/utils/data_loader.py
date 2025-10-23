import numpy as np
import torch
import random
import os
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict


class MRI_Dataset(Dataset):
    """ initial Dataset for MRI Patches """
    def __init__(self, data_path, label_path):

        data = np.load(data_path)  # (N, 16, 16)
        self.data = np.load(data_path)
        self.labels = np.load(label_path)  # (N,)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]
        img = np.transpose(img, (2, 0, 1)) # transfer to (C, H, W)
        img = torch.tensor(img, dtype=torch.float)
        label = int(self.labels[idx])  
        return img, label



class val_MRI_Dataset(Dataset):
    """ initial Dataset for validation MRI Patches """
    def __init__(self, data_path, label_path, subject_id_path):
        data = np.load(data_path)
        labels = np.load(label_path)  
        subject_ids = np.load(subject_id_path) 
        self.data = data 
        self.labels = labels
        self.subject_id = subject_ids
        
    def __len__(self):
        return len(self.data)
     
    def __getitem__(self, idx):
        img = self.data[idx]
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float)        
        label = int(self.labels[idx])  
        subject_id = self.subject_id[idx]
            
                
        return img, label, subject_id
    

class Muliti_MRI_Dataset(Dataset):
    def __init__(self, data_paths, label_paths,subject_id_paths):
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.subject_paths = subject_id_paths 
        self.data = []
        self.labels = []
        self.subject_id = []
        self.index_map = []

        for i, (data_path, lable_path,subject_id_path) in enumerate(zip(data_paths, label_paths, subject_id_paths)):
            data_array = np.load(data_path, mmap_mode='r')
            lable_array = np.load(lable_path, mmap_mode='r')
            subject_id_array = np.load(subject_id_path, mmap_mode='r')
            assert len(data_array) == len(lable_array)
            self.data.append(data_array)
            # self.labels.append(lable_array)
            self.labels.extend(lable_array.tolist())
            self.index_map.extend([(i, j) for j in range(len(data_array))])
            self.subject_id.append(subject_id_array)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        img = self.data[file_idx][sample_idx]
        img = np.transpose(img, (2, 0, 1))  # transfer to (C, H, W) 
        img = torch.tensor(img, dtype=torch.float)
        label = int(self.labels[idx])
        subject_id = self.subject_id[file_idx][sample_idx]

        return img, label, subject_id


class Muliti_val_Dataset(Dataset):
    def __init__(self, data_paths, label_paths, subject_id_paths):
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.subject_id_paths = subject_id_paths
        self.data = []
        self.labels = []
        self.subject_id = []
        self.index_map = []
        for i, (data_path, lable_path, subject_id_path) in enumerate(zip(data_paths, label_paths, subject_id_paths)):
            data_array = np.load(data_path, mmap_mode='r')
            lable_array = np.load(lable_path, mmap_mode='r')
            subject_id_array = np.load(subject_id_path, mmap_mode='r')
            assert len(data_array) == len(lable_array) == len(subject_id_array)
            self.data.append(data_array)
            self.labels.extend(lable_array.tolist())
            self.subject_id.append(subject_id_array)
            self.index_map.extend([(i, j) for j in range(len(data_array))])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        img = self.data[file_idx][sample_idx]
        img = np.transpose(img, (2, 0, 1))  # transfor to (C, H, W) format
        img = torch.tensor(img, dtype=torch.float)
        label = int(self.labels[idx])
        subject_id = self.subject_id[file_idx][sample_idx]

        return img, label, subject_id




class Muliti_test_Dataset(Dataset):
    def __init__(self, data_paths, label_paths, subject_id_paths,positions_paths):
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.subject_id_paths = subject_id_paths
        self.positions_paths = positions_paths
        self.data = []
        self.labels = []
        self.subject_id = []
        self.positions = []
        self.index_map = []

        for i, (data_path, lable_path, subject_id_path,position_path) in enumerate(zip(data_paths, label_paths, subject_id_paths,positions_paths)):
            data_array = np.load(data_path, mmap_mode='r')
            lable_array = np.load(lable_path, mmap_mode='r')
            subject_id_array = np.load(subject_id_path, mmap_mode='r')

            positions_array = np.load(positions_paths[i], mmap_mode='r')

            assert len(data_array) == len(lable_array) == len(subject_id_array)
            self.data.append(data_array)
            # self.labels.append(lable_array)
            self.labels.extend(lable_array.tolist())
            self.subject_id.append(subject_id_array)
            self.positions.append(positions_array)
            self.index_map.extend([(i, j) for j in range(len(data_array))])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        img = self.data[file_idx][sample_idx]
        img = np.transpose(img, (2, 0, 1))  # 转换为 (C, H, W) 格式
        img = torch.tensor(img, dtype=torch.float)
        label = int(self.labels[idx])
        subject_id = self.subject_id[file_idx][sample_idx]
        position = self.positions[file_idx][sample_idx]
        position = np.array(position)

        return img, label, subject_id, position


 
# Balanced for both class and folder source 
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        initial the sampler to keep the balance of classify in each batch 
        :param dataset: must include labels attribute.
        :param batch_size: size of each batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # 
        self.grouped_indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            self.grouped_indices[label].append(idx)

        # collect all labels
        self.labels = list(self.grouped_indices.keys())

    def __iter__(self):
        batch = []
        while True:
            for label in self.labels:
                if self.grouped_indices[label]:
                    idx = random.choice(self.grouped_indices[label])
                    batch.append(idx)
                    self.grouped_indices[label].remove(idx)

                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []

            # final batch 
            if len(batch) > 0:
                remaining_indices = [
                    idx for label in self.labels for idx in self.grouped_indices[label]
                ]
                if remaining_indices:
                    while len(batch) < self.batch_size and remaining_indices:
                        idx = random.choice(remaining_indices)
                        batch.append(idx)
                        remaining_indices.remove(idx)
                    yield batch
                break

            # check if all classes are exhausted
            if not any(self.grouped_indices[label] for label in self.labels):
                break

    def __len__(self):
  
        total_samples = sum(len(indices) for indices in self.grouped_indices.values())
        return total_samples // self.batch_size
    



class BalancedBatchSampler_FullEpoch(Sampler):
    def __init__(self, dataset, batch_size):
        """
        Balanced sampler across full dataset per epoch with class balance in each batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        ######self.labels = dataset.labels

        self.labels = list(dataset.labels)
        # Group indices by class
        self.class_to_indices = defaultdict(list)

        # 分组样本索引
        # self.grouped_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
        #     self.grouped_indices[label].append(idx)

        # self.classes = list(self.grouped_indices.keys())
            self.class_to_indices[label].append(idx)
        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)
        assert batch_size % self.num_classes == 0, "Batch size must be divisible by number of classes."
        self.samples_per_class = batch_size // self.num_classes
         # Determine maximum class size for oversampling
        self.max_class_len = max(len(idxs) for idxs in self.class_to_indices.values())

    def __iter__(self):

        sampled_indices = {}
        for label, idxs in self.class_to_indices.items():
            if len(idxs) < self.max_class_len:
                sampled_indices[label] = random.choices(idxs, k=self.max_class_len)
            else:
                sampled_indices[label] = random.sample(idxs, self.max_class_len)
        # Build balanced batches
        batches = []
        num_batches = self.max_class_len // self.samples_per_class
        for i in range(num_batches):
            batch = []
            for label in self.classes:
                start = i * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(sampled_indices[label][start:end])
            random.shuffle(batch)
            batches.append(batch)
        random.shuffle(batches)
        for batch in batches:
            yield batch
    def __len__(self):
            # Return number of batches per epoch
        return  self.max_class_len // self.samples_per_class
    
class BalancedBatchSampler_StopWhenExhausted(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.labels = dataset.labels
        self.batch_size = batch_size

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)
        assert batch_size % self.num_classes == 0, "Batch size must be divisible by number of classes"
        self.samples_per_class = batch_size // self.num_classes

    def __iter__(self):
        # 每类打乱索引
        class_indices = {
            label: random.sample(idxs, len(idxs))
            for label, idxs in self.class_to_indices.items()
        }

        # 最小类决定可用的 batch 数
        batch_count = min(
            len(idxs) for idxs in class_indices.values()
        ) // self.samples_per_class

        for i in range(batch_count):
            batch = []
            for label in self.classes:
                start = i * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(class_indices[label][start:end])
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return min(
            len(idxs) for idxs in self.class_to_indices.values()
        ) // self.samples_per_class




class  New_BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        支持重复采样，每个 batch 类别严格平衡，epoch 大小由最小类别样本数决定。
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = dataset.labels

        # 按类别索引分组
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)
        assert batch_size % self.num_classes == 0, "batch_size 必须能被类别数整除"

        self.samples_per_class = batch_size // self.num_classes
        self.min_class_len = min(len(v) for v in self.class_to_indices.values())
        self.num_batches_per_epoch = self.min_class_len // self.samples_per_class

    def __iter__(self):
        for _ in range(self.num_batches_per_epoch):
            batch = []
            for label in self.classes:
                # 允许重复采样
                chosen = random.choices(self.class_to_indices[label], k=self.samples_per_class)
                batch.extend(chosen)
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches_per_epoch
    



class BalancedSubjectBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, subjects_per_class=2, rounds=20):
        """
        在每个 batch 中，每个类别采样 subjects_per_class 个 subject，每个 subject 均匀采样 patch。
        :param dataset: 数据集对象，必须包含 labels 和 subject_id 属性。
        :param batch_size: batch 大小，必须能被 num_classes * subjects_per_class 整除。
        :param subjects_per_class: 每个类别每 batch 采样的 subject 数。
        :param rounds: 采样轮数。
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.subjects_per_class = subjects_per_class
        self.rounds = rounds

        self.label_to_subjects = defaultdict(set)
        self.subject_to_indices = defaultdict(list)
        self.subject_to_label = {}

        # for idx, (label, subject_id) in enumerate(zip(dataset.labels, dataset.subject_id)):
        #     self.label_to_subjects[label].add(subject_id)
        #     self.subject_to_indices[subject_id].append(idx)
        #     self.subject_to_label[subject_id] = label
        for idx, (label, subject_id) in enumerate(zip(dataset.labels, dataset.subject_id)):
            label = int(label) if isinstance(label, np.generic) else int(label)
            # 确保 subject_id 是可 hash 的
            if isinstance(subject_id, (np.ndarray, np.memmap)):
                subject_id = str(subject_id.tolist()) if subject_id.ndim > 0 else str(subject_id.item())
            else:
                subject_id = str(subject_id)
            self.label_to_subjects[label].add(subject_id)
            self.subject_to_indices[subject_id].append(idx)
            self.subject_to_label[subject_id] = label

        self.classes = sorted(self.label_to_subjects.keys())
        self.num_classes = len(self.classes)
        assert batch_size % (self.num_classes * subjects_per_class) == 0, \
            "Batch size must be divisible by num_classes * subjects_per_class"
        self.samples_per_subject = batch_size // (self.num_classes * subjects_per_class)

    def __iter__(self):
        batches = []
        used_subjects = set()
        for _ in range(self.rounds):
            # 可用 subject 列表（未被用过的）
            class_subjects = {label: list(subjects - used_subjects) for label, subjects in self.label_to_subjects.items()}
            for label in class_subjects:
                random.shuffle(class_subjects[label])
            # 能采样的 batch 数由每类可用 subject 数决定
            min_available = min(len(subs) for subs in class_subjects.values())
            num_batches = min_available // self.subjects_per_class
            for i in range(num_batches):
                batch = []
                for label in self.classes:
                    selected_subjects = class_subjects[label][i * self.subjects_per_class : (i + 1) * self.subjects_per_class]
                    for subject_id in selected_subjects:
                        used_subjects.add(subject_id)
                        indices = self.subject_to_indices[subject_id]
                        if len(indices) >= self.samples_per_subject:
                            batch.extend(random.sample(indices, self.samples_per_subject))
                        else:
                            batch.extend(random.choices(indices, k=self.samples_per_subject))
                random.shuffle(batch)
                batches.append(batch)
        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        min_subjects = min(len(subjects) for subjects in self.label_to_subjects.values())
        batches_per_round = (min_subjects // self.subjects_per_class)
        return self.rounds * batches_per_round
