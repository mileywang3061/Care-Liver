import glob
import numpy as np
import os
import shutil
import nibabel as nib
from scipy.ndimage import zoom


def padding_crop(data, target_shape):
    """
    Pads or crops the data to match the target shape.
    """
    current_shape = data.shape
    padding = [(max(0, (target_shape[i] - current_shape[i]) // 2), 
                max(0, (target_shape[i] - current_shape[i] + 1) // 2)) for i in range(len(current_shape))]
    cropped_data = np.pad(data, padding, mode='constant', constant_values=0)
    
    if cropped_data.shape != target_shape:
        excess = [(cropped_data.shape[i] - target_shape[i]) // 2 for i in range(len(target_shape))]
        slices = tuple(slice(excess[i], excess[i] + target_shape[i]) for i in range(len(target_shape)))
        cropped_data = cropped_data[slices]
    
    return cropped_data

def reverse_padding_crop(data, original_shape):
    """
    Reverses the padding_crop operation to restore the original shape.
    """
    current_shape = data.shape
    padding = [(max(0, (original_shape[i] - current_shape[i]) // 2), 
                max(0, (original_shape[i] - current_shape[i] + 1) // 2)) for i in range(len(current_shape))]
    
    unpadded_data = np.pad(data, padding, mode='constant', constant_values=0)
    
    if unpadded_data.shape != original_shape:
        excess = [(unpadded_data.shape[i] - original_shape[i]) // 2 for i in range(len(original_shape))]
        slices = tuple(slice(excess[i], excess[i] + original_shape[i]) for i in range(len(original_shape)))
        unpadded_data = unpadded_data[slices]
    
    return unpadded_data

def preprocess(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_list = glob.glob(input_dir + '/Vendor*/**/*.nii.gz')
    data_list = [x for x in data_list if 'mask' not in x]
    data_list = [x for x in data_list if 'GED1' not in x]
    data_list = [x for x in data_list if 'GED2' not in x]
    data_list = [x for x in data_list if 'GED3' not in x]
    target_physical_size = [1,1,2.5]
    crop_shape = (400, 400, 80) 
    small_shape = (256, 256, 48) 

    shapes_img = []

    for i, ged4_path in enumerate(data_list):
        fid = os.path.basename(os.path.dirname(ged4_path))

        ftype = os.path.basename(ged4_path).split('.nii')[0]
        
        
        img_nii = nib.load(ged4_path)
        physical_shape = img_nii.header.get_zooms()
        zoom_factor = np.array(physical_shape[:3]) / np.array(target_physical_size)
        target_shape = list(np.round(np.array(img_nii.shape)[:3] * zoom_factor).astype(int))
        if target_shape not in shapes_img:
            shapes_img.append(target_shape)
        img = img_nii.get_fdata()
        if len(img.shape) == 4:
            img = img.squeeze()
        img_reshaped = zoom(img, zoom_factor, order=1).astype(np.float64)
        
        new_affine = img_nii.affine.copy()
        rounded = np.round(new_affine[:3, :3], 8)
        signs = np.sign(rounded)
        new_affine[:3, :3] = signs * [[1,0,0],[0,1,0],[0,0,2.5]]

        cropped_img = padding_crop(img_reshaped, crop_shape).astype(np.float32)

        samll_zoom_factor = np.array(small_shape)/ np.array(cropped_img.shape)
        img_small = zoom(cropped_img, samll_zoom_factor, order=1).astype(np.float32)
        
        nib.save(nib.Nifti1Image(img_small, new_affine), f'{output_dir}/{fid}_{ftype}.nii.gz')

        print(f'Preprocessing - {i+1}/{len(data_list)}: {fid}_{ftype} - {img.shape} -> {img_reshaped.shape} -> {cropped_img.shape} -> {img_small.shape}')
        
def postprocess(org_dir, pred_dir, output_dir='/output/LiSeg_pred/'):

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_list = glob.glob(org_dir + 'Vendor*/**/*.nii.gz')
    data_list = [x for x in data_list if 'mask' not in x]
    data_list = [x for x in data_list if 'GED1' not in x]
    data_list = [x for x in data_list if 'GED2' not in x]
    data_list = [x for x in data_list if 'GED3' not in x]
    target_physical_size = [1,1,2.5]
    crop_shape = (400, 400, 80) 

    for i, org_path in enumerate(data_list):
        fid = os.path.basename(os.path.dirname(org_path))
        ftype = os.path.basename(org_path).split('.nii')[0]
        mask_name = fid + f'_{ftype}_mask.nii.gz'
        mask_path = pred_dir + '/' + mask_name
        if not os.path.exists(mask_path):
            print(f'Mask not found for {fid}: {mask_path}')
            continue

        org_nii = nib.load(org_path)

        # print(f'{i+1}/{len(data_list)}: {fid}')
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata()
        zoom_factor = np.array(crop_shape)/ np.array(mask.shape)

        mask_reshaped = zoom(mask, zoom_factor, order=0).astype(np.uint8)

        physical_shape = org_nii.header.get_zooms()
        zoom_factor = np.array(physical_shape[:3]) / np.array(target_physical_size)
        target_shape = list(np.round(np.array(org_nii.shape)[:3] * zoom_factor).astype(int))

        mask_reversed = reverse_padding_crop(mask_reshaped, target_shape)
        
        assert list(mask_reversed.shape) == target_shape

        zoom_factor = np.array(org_nii.shape[:3]) / np.array(mask_reversed.shape)
        mask_org_shape = zoom(mask_reversed, zoom_factor, order=0).astype(np.uint8)
        if len(org_nii.shape) == 4:
            mask_org_shape = mask_org_shape[..., np.newaxis]
        output_name = f'{output_dir}/{fid}/{ftype}_pred.nii.gz'
        if not os.path.exists(os.path.dirname(output_name)):
            os.makedirs(os.path.dirname(output_name))
        nib.save(nib.Nifti1Image(mask_org_shape, org_nii.affine), output_name)
        print(f'Postprocessing - {i+1}/{len(data_list)}: {fid} - {mask_nii.shape} -> {mask_reshaped.shape} -> {mask_reversed.shape} -> {org_nii.shape}')
        