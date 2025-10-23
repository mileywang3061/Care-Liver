import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
from sklearn.model_selection import StratifiedKFold
from scipy.ndimage import zoom




def classes_management(class_number,category):
    if class_number == 4:
        if category == '1':
            new_category = 0
        elif category == '2':
            new_category = 2
        elif category == '3':
            new_category = 3
        elif category == '4':
            new_category = 1

    elif class_number == 2:
        if category == '1':
            new_category = 0
        else:
            new_category = 1
    elif class_number == 1:
        if category == '4':
            new_category = 0
        else:
            new_category = 1

    return new_category



def get_class(class_number):
    folder_file = args.data_path  # data path 
    mask_file = args.mask_path  # mask path

    category_dict = defaultdict(list)  # {label: [{'filename': ..., 'mri_path': [...], 'mask_path': ...}, ...]}

    for vendor in ['Vendor_A', 'Vendor_B1', 'Vendor_B2']:
        vendor_path = os.path.join(folder_file, vendor)
        if not os.path.isdir(vendor_path):
            continue
    
        subject_dirs = [d for d in os.listdir(vendor_path) if os.path.isdir(os.path.join(vendor_path, d))]
        for subject in subject_dirs:
            subject_path = os.path.join(vendor_path, subject)
            file_name = subject
            category = file_name.split('-')[-1][-1]  # 提取类别信息
            file_path = os.path.join(subject_path, file_name)  
            mask_path = os.path.join(mask_file, vendor, 'ged4', file_name+'_resize.nii.gz')  # 假设mask文件名与文件夹名相同

            order = ['T1','T2','DWI']
            nii_files = []
            for k in order:
                match = glob.glob(os.path.join(subject_path, f'*{k}*.nii.gz'))
                if len(match) > 0:
                    nii_files.append(match[0])
                else:
                    nii_files.append('0')

            category = classes_management(class_number, category)

            category_dict[category].append({
                    'filename': file_name,
                    'mri_path': nii_files,
                    'mask_path': mask_path
                     })
    return category_dict



def generate_dataset(category_dic):
    classes = []
    num_samples = []
    for category, count in category_dic.items():
        print(f"Category: {category}, Count: {count}")  
        classes.append(category)
        num_samples.append(len(count))

    total_subjects = sum(num_samples)  # 120
    num_test_samples = np.round(np.array(num_samples) * 0.1).astype(int)  # 四舍五入
    num_train_val_samples = np.array(num_samples) - num_test_samples  

    return 
      

def split_data(data_dict):
    train_val_data = {}
    test_data = {} 
    inference_data ={} 
    for category, items in data_dict.items():

        if category in [0,1]:
            test_size = round(0.1 * len(items))
            train_val_items, test_items = train_test_split(items, test_size=test_size, random_state=42)
            train_val_data[category] = train_val_items
            test_data[category] = test_items

        else: 
            test_data[category] = items

    return train_val_data, test_data,inference_data

def expand_data_for_balance(train_val_data):

    balanced_data = defaultdict(list)
    
    max_samples = max(len(items) for items in train_val_data.values())
    for category, items in train_val_data.items():
        current_samples = len(items)

        if current_samples < max_samples:
            num_to_add = max_samples - current_samples
            additional_items = random.choices(items, k=num_to_add)
            balanced_data[category].extend(items + additional_items)
        else:
            balanced_data[category].extend(items)

    
    return balanced_data


def create_kfolds(extended_data, n_splits=4, shuffle=True):
    data = []
    labels =[]
    for category, items in extended_data.items():
        for item in items:
            data.append(item)
            labels.append(category)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    Kfolds = []
    for train_index, val_index in skf.split(data,labels):
        train_data = [data[i] for i in train_index]
        val_data = [data[i] for i in val_index]
        train_labels = [labels[i] for i in train_index]
        val_labels = [labels[i] for i in val_index]
        if shuffle:
            combined_train = list(zip(train_data, train_labels))
            combined_val = list(zip(val_data, val_labels))
            random.shuffle(combined_train)
            random.shuffle(combined_val)
            train_data, train_labels = zip(*combined_train)
            val_data, val_labels = zip(*combined_val)
       

        Kfolds.append({"train": train_data, "val": val_data, "train_labels": train_labels, "val_labels": val_labels})
       
    return Kfolds



def create_test_data(train_val_data, test_data, inference_data):
    """
    创建一个包含所有数据的字典，格式与 test_data 保持一致。
    :param train_val_data: 训练和验证数据字典
    :param test_data: 测试数据字典
    :param inference_data: 推理数据字典
    :return: 包含所有数据的字典 all_data
    """
    all_test_data = {}

    # 合并 test_data
    for category, items in test_data.items():
        if category not in all_test_data:
            all_test_data[category] = []
        all_test_data[category].extend(items)

    # 合并 inference_data
    for category, items in inference_data.items():
        if category not in all_test_data:
           all_test_data[category] = []
        all_test_data[category].extend(items)

    return all_test_data
            
