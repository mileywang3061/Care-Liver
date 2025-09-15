from os.path import join
from os import listdir
from scipy.io import loadmat
import SimpleITK as sitk
import pandas as pd
from torch.utils import data
import numpy as np
import glob
import scipy.io as sio
import os
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
data_labeled = glob.glob(data_path + '/labeled_1_1_25mm_cropped_256_256_48/*GED4.nii.gz')
data_labeled_gong = glob.glob(data_path + '/gong_256/*.nii.gz')
data_labeled_gong = [f for f in data_labeled_gong if 'mask' not in f]  # Exclude labeled files
data_labeled = data_labeled + data_labeled_gong
np.random.seed(42)
np.random.shuffle(data_labeled)
data_labeled_train = data_labeled[:-10]
data_labeled_test = data_labeled[-10:]
# data_unlabeled = glob.glob(data_path + '/unlabeled_1_1_25mm_cropped_256_256_48/*.nii.gz')
data_unlabeled = glob.glob('E:/OneDrive - The University of Nottingham/Projects/Data/_DB_Liver/LiQA_val/256/*.nii.gz')
data_unlabeled = [f for f in data_unlabeled if 'GED1' not in f]  # Exclude labeled files
data_unlabeled = [f for f in data_unlabeled if 'GED2' not in f] 
data_unlabeled = [f for f in data_unlabeled if 'GED3' not in f] 

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
        self.labeled_filenames = data_unlabeled
        # self.labeled_filenames = data_labeled_test
        print(self.labeled_filenames)

    def __getitem__(self, index):
        labed_img = nib.load(self.labeled_filenames[index])
        affine = labed_img.affine
        labed_img = apply_affine(labed_img.get_fdata(), labed_img.affine)
        labed_img = np.transpose(labed_img, (2, 1, 0))  # Change to (z, y, x)
        labed_img = labed_img.astype(np.float32)
        labed_img = labed_img[np.newaxis, :, :, :]
        labed_img = zero_mean_normalization(labed_img)
        label_name = self.labeled_filenames[index].replace('.nii.gz', '_mask.nii.gz')
        labed_lab = np.zeros_like(labed_img, dtype=np.float32)
        if '_mask' in label_name and os.path.exists(label_name):
            labed_lab = nib.load(label_name)
            labed_lab = apply_affine(labed_lab.get_fdata(), labed_lab.affine)
            labed_lab = np.transpose(labed_lab, (2, 1, 0))  # Change to (z, y, x)
            labed_lab = self.to_categorical(labed_lab, self.num_classes)
            labed_lab = labed_lab.astype(np.float32)

        return labed_img, labed_lab, self.labeled_filenames[index], affine

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
        return len(self.labeled_filenames)

