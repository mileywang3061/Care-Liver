from os.path import join
from os import listdir
from scipy.io import loadmat
import SimpleITK as sitk
import pandas as pd
from torch.utils import data
import numpy as np
import glob
import scipy.io as sio
import nibabel as nib

def apply_affine(datain, affine):
    data = datain.copy()
    flip_axes = [affine[i, i] < 0 for i in range(3)]
    numpy_axes = [2, 1, 0]
    for flip, axis in zip(flip_axes, numpy_axes):
        if flip:
            data = np.flip(data, axis=axis)
    
    return data

# from utils.augmentation_cpu import MirrorTransform, SpatialTransform

data_path = 'E:/Data/_DB-Liver/LiQA_training_data/processed'
# data_labeled = glob.glob(data_path + '/labeled_1_1_25mm_cropped_256_256_48/*GED4.nii.gz')
# data_labeled_gong = glob.glob(data_path + '/gong_256/*.nii.gz')
# data_labeled_gong = [f for f in data_labeled_gong if 'mask' not in f]  # Exclude labeled files
# data_labeled = data_labeled + data_labeled_gong
# np.random.seed(42)
# np.random.shuffle(data_labeled)
# data_labeled_train = data_labeled[:20]
# data_labeled_test = data_labeled[20:]
# data_unlabeled = glob.glob(data_path + '/unlabeled_1_1_25mm_cropped_256_256_48/*.nii.gz')
# data_unlabeled = [f for f in data_unlabeled if 'GED1' not in f and 'GED2' not in f and 'GED3' not in f] 

# Train: All to 4
# data_source = glob.glob(data_path + '/*_1_1_25mm_cropped_256_256_48/*.nii.gz')
# data_source = [f for f in data_source if 'GED4' not in f] 
# data_target = glob.glob(data_path + '/*_1_1_25mm_cropped_256_256_48/*GED4.nii.gz')

# data_source = [f for f in data_source if 'mask' not in f]
# data_target = [f for f in data_target if 'mask' not in f]

# import os
# tmp = []
# for d in data_source:
#     idx = int(os.path.basename(d)[:4])
#     if idx >= 1074:
#         tmp.append(d)
# data_source = tmp


# Val: All to 4
data_source = glob.glob('E:/OneDrive - The University of Nottingham/Projects/Data/_DB_Liver/LiQA_val/256/*.nii.gz')
data_source = [f for f in data_source if 'GED4' not in f] 
data_target = glob.glob('E:/OneDrive - The University of Nottingham/Projects/Data/_DB_Liver/LiQA_val/256/*GED4.nii.gz')

import os
tmp = []
for d in data_source:
    idx = int(os.path.basename(d)[:4])
    if idx >= 985:
        tmp.append(d)
data_source = tmp

def zero_mean_normalization(image):
    mean = np.mean(image)
    std = np.std(image)
    if std == 0:
        return image - mean
    else:
        return (image - mean) / std

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii"])

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, labeled_file_dir, num_classes, shot=5):
        super(DatasetFromFolder3D, self).__init__()
        self.num_classes = num_classes
        # self.labeled_filenames = data_labeled
        self.source = data_source
        self.target = data_target
        self.total = len(self.source)

    def __getitem__(self, index):
        source_path = self.source[index]
        target_path = None
        other_suffix = ['DWI_800', 'T1', 'T2', 'GED1', 'GED2', 'GED3', 'DCE_800']
        for suffix in other_suffix:
            if suffix in source_path:
                target_path = source_path.replace(suffix, 'GED4')
        if target_path not in self.target:
            target_path = target_path.replace('unlabeled_1_1_25mm_cropped_256_256_48', 'labeled_1_1_25mm_cropped_256_256_48')
            assert target_path in self.target, f"Target path {target_path} not found in target list."

        labed_img1 = nib.load(source_path)
        affine1 = labed_img1.affine
        labed_img1 = apply_affine(labed_img1.get_fdata(), labed_img1.affine)
        labed_img1 = np.transpose(labed_img1, (2, 1, 0))  # Change to (z, y, x)
        labed_img1 = labed_img1.astype(np.float32)
        labed_img1 = labed_img1[np.newaxis, :, :, :]
        labed_img1 = zero_mean_normalization(labed_img1)
        labed_lab1 = None
        # labed_lab1 = nib.load(self.source[index//len(self.target)].replace('.nii.gz', '_mask.nii.gz'))
        # labed_lab1 = apply_affine(labed_lab1.get_fdata(), labed_lab1.affine)
        # labed_lab1 = np.transpose(labed_lab1, (2, 1, 0))  # Change to (z, y, x)
        # labed_lab1 = self.to_categorical(labed_lab1, self.num_classes)
        # labed_lab1 = labed_lab1.astype(np.float32)

        labed_img2 = nib.load(target_path)
        affine2 = labed_img2.affine
        labed_img2 = apply_affine(labed_img2.get_fdata(), labed_img2.affine)
        labed_img2 = np.transpose(labed_img2, (2, 1, 0))  # Change to (z, y, x)
        labed_img2 = labed_img2.astype(np.float32)
        labed_img2 = labed_img2[np.newaxis, :, :, :]
        labed_img2 = zero_mean_normalization(labed_img2)
        labed_lab2 = None
        # labed_lab2 = nib.load(self.target[index % len(self.target)].replace('.nii.gz', '_mask.nii.gz'))
        # labed_lab2 = apply_affine(labed_lab2.get_fdata(), labed_lab2.affine)
        # labed_lab2 = np.transpose(labed_lab2, (2, 1, 0))  # Change to (z, y, x)
        # labed_lab2 = self.to_categorical(labed_lab2, self.num_classes)
        # labed_lab2 = labed_lab2.astype(np.float32)

        return labed_img1, labed_img2, \
               source_path, target_path, affine1, affine2
    
        # return labed_img1, labed_lab1, labed_img2, labed_lab2, \
        #        self.source[index//len(self.target)], self.target[index % len(self.target)], affine1, affine2

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.source) * len(self.target)

