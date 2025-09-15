from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import numpy as np
import glob
import os
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

data_path = 'E:/Data/_DB-Liver/LiQA_training_data/processed'
data_care = glob.glob(data_path + '/labeled_1_1_25mm_cropped_256_256_48/*GED4.nii.gz')
data_labeled = data_care[:25]
data_addition = glob.glob('E:/Data/_DB-Liver/CirrMRI600+/forcare_256_crop/*.nii.gz')
data_addition = [f for f in data_addition if 'mask' not in f]  # Exclude labeled files
# data_addition = [f for f in data_addition if 'test_' not in f]  # Exclude labeled files
data_labeled = data_labeled + data_addition
# data_labeled_gong = glob.glob(data_path + '/gong_256/*.nii.gz')
# data_labeled_gong = [f for f in data_labeled_gong if 'mask' not in f]  # Exclude labeled files
data_labeled = data_labeled
np.random.seed(42)
np.random.shuffle(data_labeled)
data_labeled_train = data_labeled
# data_labeled_test = glob.glob('E:/Data/_DB-Liver/CirrMRI600+/forcare_256_crop/test_images*.nii.gz')
data_labeled_test = data_care[25:]
data_labeled_test = [f for f in data_labeled_test if 'mask' not in f]  # Exclude labeled files
data_unlabeled = glob.glob(data_path + '/unlabeled_1_1_25mm_cropped_256_256_48/*.nii.gz')
# data_unlabeled = [f for f in data_unlabeled if 'GED1' not in f and 'GED2' not in f and 'GED3' not in f]  # Exclude labeled files


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
    def __init__(self, labeled_file_dir, unlabeled_file_dir, num_classes, shot=5):
        super(DatasetFromFolder3D, self).__init__()
        self.num_classes = num_classes
        self.labeled_filenames = data_labeled_train
        self.unlabeled_filenames = data_unlabeled

    def __getitem__(self, index):
        random_index = np.random.randint(low=0, high=len(self.labeled_filenames))
        # labed_img = sitk.ReadImage(self.labeled_filenames[random_index])
        # labed_img = sitk.GetArrayFromImage(labed_img)
        labed_img = nib.load(self.labeled_filenames[random_index])
        labed_img = apply_affine(labed_img.get_fdata(), labed_img.affine)
        labed_img = np.transpose(labed_img, (2, 1, 0))  # Change to (z, y, x)
        labed_img = labed_img.astype(np.float32)
        labed_img = labed_img[np.newaxis, :, :, :]
        labed_img = zero_mean_normalization(labed_img)
        # labed_lab = sitk.ReadImage(self.labeled_filenames[random_index].replace('GED4.nii.gz', 'GED4_mask.nii.gz'))
        # labed_lab = sitk.GetArrayFromImage(labed_lab)
        labed_lab = nib.load(self.labeled_filenames[random_index].replace('.nii.gz', '_mask.nii.gz'))
        labed_lab = apply_affine(labed_lab.get_fdata(), labed_lab.affine)
        labed_lab = np.transpose(labed_lab, (2, 1, 0))  # Change to (z, y, x)
        labed_lab = self.to_categorical(labed_lab, self.num_classes)
        labed_lab = labed_lab.astype(np.float32)

        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        # unlabed_img1 = sitk.ReadImage(self.unlabeled_filenames[random_index])
        # unlabed_img1 = sitk.GetArrayFromImage(unlabed_img1)
        unlabed_img1 = nib.load(self.unlabeled_filenames[random_index])
        unlabed_img1 = apply_affine(unlabed_img1.get_fdata(), unlabed_img1.affine)
        unlabed_img1 = np.transpose(unlabed_img1, (2, 1, 0))  # Change to (z, y, x)
        unlabed_img1 = unlabed_img1.astype(np.float32)
        unlabed_img1 = unlabed_img1[np.newaxis, :, :, :]
        unlabed_img1 = zero_mean_normalization(unlabed_img1)

        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        # unlabed_img2 = sitk.ReadImage(self.unlabeled_filenames[random_index])
        # unlabed_img2 = sitk.GetArrayFromImage(unlabed_img2)
        unlabed_img2 = nib.load(self.unlabeled_filenames[random_index])
        unlabed_img2 = apply_affine(unlabed_img2.get_fdata(), unlabed_img2.affine)
        unlabed_img2 = np.transpose(unlabed_img2, (2, 1, 0))
        unlabed_img2 = unlabed_img2.astype(np.float32)
        unlabed_img2 = unlabed_img2[np.newaxis, :, :, :]
        unlabed_img2 = zero_mean_normalization(unlabed_img2)

        return labed_img, labed_lab, unlabed_img1, unlabed_img2

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
        return len(self.unlabeled_filenames)+len(self.labeled_filenames)

