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


import logging

# ------------- Logging -------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# ---------- Safe numeric helpers ----------
def _safe_std(x: np.ndarray, eps: float = 1e-6) -> float:
    """Return a non-zero std to avoid division-by-zero."""
    try:
        s = float(np.std(x)) if x.size else 0.0
    except Exception:
        s = 0.0
    return s if s > eps else eps



def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class(folder_file, mask_file, data_type):

    file_dict = defaultdict(list)  
    for vendor in ['Vendor_A', 'Vendor_B1', 'Vendor_B2','Vendor_C']:
        vendor_path = os.path.join(folder_file, vendor)
        if not os.path.isdir(vendor_path):
            continue
    
        subject_dirs = [d for d in os.listdir(vendor_path) if os.path.isdir(os.path.join(vendor_path, d))]
        for subject in subject_dirs:
            subject_path = os.path.join(vendor_path, subject)

            file_name = subject
            file_path = subject_path  
            mask_path = os.path.join(mask_file, file_name, 'GED4_pred.nii.gz')

            # Collect nii files by modality; fix glob pattern and add checks
            if data_type == 'NonContrast':
                order = ['T1','T2','DWI']
                nii_files = []
                for k in order:
                    match = glob.glob(os.path.join(subject_path, f'*{k}*.nii.gz'))
                    if len(match) > 0:
                        nii_files.append(match[0])
                    else:
                        nii_files.append('0')
                
                # print(f"nii_files for {file_name}: {nii_files}", flush=True)
                # print('len of nii_files:', len(nii_files), flush=True)
            else:
                # NOTE: original code used '.nii.gz' without '*', which returns empty
                # nii_files = glob.glob(os.path.join(subject_path, '*.nii.gz'))
                order = ['T1','T2','DWI','GED1','GED2','GED3','GED4']
                nii_files = []
                for k in order:
                    match = glob.glob(os.path.join(subject_path, f'*{k}*.nii.gz'))
                    if len(match) > 0:
                        nii_files.append(match[0])
                    else:
                        nii_files.append('0')
                
                # print('nii_files', nii_files, flush=True)
                # print('len(nii_files)', len(nii_files), flush=True)


            if not os.path.exists(mask_path):
                logging.warning(f'Mask not found for case {file_name}: {mask_path}. Skipping.')
                continue
            if len(nii_files) == 0:
                logging.warning(f'No NIfTI files for case {file_name} in {subject_path}. Skipping.')
                continue

            file_dict[file_name].append({
                'filename': file_name,
                'mri_path': nii_files,
                'mask_path': mask_path
            })
            
    return file_dict


def sitk_resize_image(img, target_spacing=(1.0, 1.0, 2.5), interp=sitk.sitkLinear):

    if isinstance(img, nib.Nifti1Image):
        data = img.get_fdata()
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))  # 转换为 [Z, Y, X] 顺序
        elif data.ndim == 4:
            data = data[..., 0]  # 取出第一个通道或时间点，结果为3D
            data = np.transpose(data, (2, 1, 0))  # 再转换为 [Z, Y, X] 顺序

        img = sitk.GetImageFromArray(data)
    elif isinstance(img, np.ndarray):
        data  = np.transpose(img, (2, 1, 0))  # 转换为 [Z, Y, X] 顺序
        img = sitk.GetImageFromArray(data)
    elif isinstance(img, sitk.Image):
        pass
    else:
        raise TypeError(f"Unsupported image type: {type(img)}. Expected nibabel or SimpleITK image.")

    if img.GetDimension() == 4:
        size = list(img.GetSize())
        size[3] = 0  # 只提取第一个 time/frame
        index = [0, 0, 0, 0]
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(size)
        extractor.SetIndex(index)
        img = extractor.Execute(img)

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputDirection(img.GetDirection())
    resample.SetInterpolator(interp)

    resampled_img = resample.Execute(img)

    return resampled_img


def safe_registration(fixed, moving, **kwargs):
    """
    Wrap ants.registration with fallbacks to avoid hard crashes:
      1) Try given params;
      2) If it fails, retry with simpler Affine;
      3) If still failing, return an identity-like output.
    """
    try:
        return ants.registration(fixed=fixed, moving=moving, **kwargs)
    except Exception as e:
        logging.warning(f'ANTs registration failed (params={kwargs}): {e}. Retrying with Affine...')
        try:
            base = dict(type_of_transform='Affine', random_seed=0, verbose=False)
            base.update({k: v for k, v in kwargs.items() if k != 'type_of_transform'})
            return ants.registration(fixed=fixed, moving=moving, **base)
        except Exception as ee:
            logging.error(f'Affine registration also failed: {ee}. Fallback to identity transform.')
            return {'fwdtransforms': [], 'invtransforms': [], 'warpedmovout': moving}
    

class MRIPatch_dataset():
    def __init__(self, mri_path, mask_path, patch_size, cover_rate,data_type,max_patches_per_slice=100):
        """
        初始化 MRI Patch 数据集。
        :param mri_path: MRI 文件路径。
        :param mask_path: Mask 文件路径。
        :param patch_size: Patch 的大小。
        :param cover_rate: 覆盖率阈值。
        :param max_patches_per_slice: 每个 slice 最大提取的 patch 数量。
        """
        self.mri_path = mri_path
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.cover_rate = cover_rate
        self.max_patches_per_slice = max_patches_per_slice
        self.data_type = data_type  

    def extract_patches(self):
        patches = []
        patch_positions = []
        extracted_rate = 0.1  
        try:
            mask_img = nli.load_img(self.mask_path)
            mask_img = sitk_resize_image(mask_img, target_spacing=(1.0, 1.0, 2.5), interp=sitk.sitkLinear)
            mask_data = sitk.GetArrayFromImage(mask_img).astype(np.float32)
            mask_data = np.nan_to_num(mask_data, nan=0.0, posinf=0.0, neginf=0.0)
            mask_data = np.transpose(mask_data, (2, 1, 0))
            # print(f"Mask shape after loading and resizing: {mask_data.shape}", flush=True)
            standard_shape = (mask_data.shape[0], mask_data.shape[1], mask_data.shape[2])
            # print(f"Standard shape set to: {standard_shape}", flush=True)
        except Exception as e:
            logging.error(f'Failed to load/prepare mask: {self.mask_path} - {e}')
            return [], []
        self.mri_path = self.mri_path[0]
    

        mri_data_list = []

        candidates = self.mri_path[0] if isinstance(self.mri_path[0], list) else self.mri_path
        valid_paths = [p for p in candidates if p != "0"]

        if len(valid_paths) == 0:
            raise ValueError(f"No valid MRI path found in {self.mri_path}")

        parent_dir = os.path.dirname(valid_paths[0])

        fixed_path = os.path.join(parent_dir, 'GED4.nii.gz')
        # print('fixed_path', fixed_path, flush=True)
        if not os.path.exists(fixed_path):
            logging.warning(f'Fixed image not found: {fixed_path}. Using first MRI as fixed.')
            fixed_path = self.mri_path[0][0] if isinstance(self.mri_path[0], list) else self.mri_path[0]

        try:
            fixed = ants.image_read(fixed_path)
        except Exception as e:
            logging.error(f'Failed to read fixed image {fixed_path}: {e}')
            return [], []

        for path_entry in self.mri_path:
            if path_entry == '0':
                mri_data = np.ones(standard_shape, dtype=np.float32)
                # print('0 shpae', mri_data.shape, flush=True)
                mri_data_list.append(mri_data)

            else:
                path = path_entry[0] if isinstance(path_entry, list) else path_entry
                if not os.path.exists(path):
                    logging.warning(f'MRI file missing: {path}. Skipping this modality.')
                    continue
                try:
                    moving = ants.image_read(path)
                except Exception as e:
                    logging.warning(f'Failed to read moving image {path}: {e}. Skipping modality.')
                    continue

                reg = safe_registration(
                    fixed=fixed,
                    moving=moving,
                    type_of_transform='Rigid',
                    reg_iterations=(1000, 500, 250, 100),
                    verbose=False,
                    random_seed=0
                )

                try:
                    reg_img = reg['warpedmovout'].numpy() if 'warpedmovout' in reg else moving.numpy()
                    mri_img = sitk_resize_image(reg_img, target_spacing=(1.0, 1.0, 2.5), interp=sitk.sitkLinear)
                    img = sitk.GetArrayFromImage(mri_img).astype(np.float32)
                    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                    img = np.transpose(img, (2, 1, 0))
                    data = img
                    # print(f"Modality {path} shape after registration and resizing: {data.shape}", flush=True)
                    if data.ndim == 4:
                        data = data[..., 0]  # use first channel if unexpectedly 4D
                    mri_data_list.append(data)
                except Exception as e:
                    logging.warning(f'Post-registration processing failed for {path}: {e}. Skipping this modality.')
                    continue

        if len(mri_data_list) == 0:
            logging.error('No valid MRI modalities after loading/registration.')
            return [], []


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

        # Sanity check
        if any(m.ndim != 3 for m in mri_data_list):
            logging.error('Unexpected modality shape (expected 3D arrays). Skipping case.')
            return [], []

        num_slices = mask_data.shape[2]
        half_size = self.patch_size // 2
        mri_z_shapes = [m.shape[2] for m in mri_data_list]
        min_z = min([num_slices] + mri_z_shapes)   

        for i in range(min_z):
            mask_slice = mask_data[:, :, i]
            pos = np.argwhere(mask_slice == 1)
            if pos.size == 0:
                continue
            positions = pos[:, :2]

            slice_patches = []
            slice_positions = []


            for x, y in positions:

                if (x - half_size >= 0 and x + half_size < mask_slice.shape[0] and
                    y - half_size >= 0 and y + half_size < mask_slice.shape[1]):

                    patch_area = mask_slice[x-half_size:x+half_size, y-half_size:y+half_size]
                    coverage = np.sum(patch_area) / (self.patch_size ** 2)
                    if coverage >= self.cover_rate:
                        patch = np.stack([m[x-half_size:x+half_size, y-half_size:y+half_size, i] for m in mri_data_list], axis=0)
                        slice_patches.append(patch)
                        slice_positions.append((i, x, y))

            if len(slice_patches) > 0:
                num_to_extract = max(1, int(len(slice_patches) * extracted_rate))  # 至少提取一个
                indices = np.random.choice(len(slice_patches), num_to_extract, replace=False)
                slice_patches = [slice_patches[j] for j in indices]
                slice_positions = [slice_positions[j] for j in indices]

            # 将当前 slice 的 patches 添加到总结果中
            patches.extend(slice_patches)
            patch_positions.extend(slice_positions)

            # 手动释放内存

            del  slice_patches, slice_positions
            gc.collect()
        

        for idx, patch in enumerate(patches):
            if 0 in patch.shape:
                print(f"[Warning] Patch {idx} shape: {patch.shape}")

        return patches, patch_positions