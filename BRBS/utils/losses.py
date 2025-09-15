import torch
import torch.nn.functional as F
import numpy as np
import math


def mi_loss(I, J, win=None, num_bins=32, sigma_ratio=1.0, minval=0.0, maxval=1.0, patch_size=5, eps=1e-6):
    """
    Local Mutual Information (non-overlapping patch based) for 3D images.
    I, J: input tensors of shape [B, 1, D, H, W]
    """
    assert I.shape[1] == 1 and J.shape[1] == 1, "Input must have shape [B, 1, D, H, W]"

    B, C, D, H, W = I.shape
    sigma = ((maxval - minval) / num_bins) * sigma_ratio
    preterm = 1 / (2 * sigma ** 2)
    bin_centers = torch.linspace(minval, maxval, num_bins, device=I.device).view(1, 1, num_bins)

    # Pad to make divisible by patch size
    D_pad = (-D) % patch_size
    H_pad = (-H) % patch_size
    W_pad = (-W) % patch_size
    pad = (W_pad//2, W_pad - W_pad//2, H_pad//2, H_pad - H_pad//2, D_pad//2, D_pad - D_pad//2)
    I = F.pad(I, pad, mode='constant', value=0)
    J = F.pad(J, pad, mode='constant', value=0)

    # Patchify
    D_new, H_new, W_new = I.shape[2:]
    I_patch = I.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
    J_patch = J.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)

    I_patch = I_patch.contiguous().view(-1, patch_size**3, 1)
    J_patch = J_patch.contiguous().view(-1, patch_size**3, 1)

    # Soft histogram
    I_hist = torch.exp(-preterm * (I_patch - bin_centers) ** 2)
    I_hist = I_hist / (I_hist.sum(dim=-1, keepdim=True) + eps)

    J_hist = torch.exp(-preterm * (J_patch - bin_centers) ** 2)
    J_hist = J_hist / (J_hist.sum(dim=-1, keepdim=True) + eps)

    # Joint histogram
    Pab = torch.bmm(I_hist.transpose(1, 2), J_hist) / patch_size ** 3
    Pa = torch.mean(I_hist, dim=1, keepdim=True)
    Pb = torch.mean(J_hist, dim=1, keepdim=True)
    Papb = torch.bmm(Pa.transpose(1, 2), Pb) + eps

    mi = torch.sum(Pab * torch.log(Pab / Papb + eps), dim=(1, 2))

    return 1-mi.mean()  # negative MI as loss

def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :]) 
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :]) 
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1]) 

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0

def app_gradient_loss(mask, s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(mask * (dx + dy + dz))
    return d / 3.0

def ncc_loss(I, J, win=None):
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)
    
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross*cross / (I_var*J_var + 1e-5)

    return 1 - torch.mean(cc)

def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = int(np.prod(win))
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

def dice_coef(y_true, y_pred):
    smooth = 1.
    a = torch.sum(y_true * y_pred, (2, 3, 4))
    b = torch.sum(y_true**2, (2, 3, 4))
    c = torch.sum(y_pred**2, (2, 3, 4))
    dice = (2 * a + smooth) / (b + c + smooth)
    return torch.mean(dice)

def dice_loss(y_true, y_pred):
    d = dice_coef(y_true, y_pred)
    return 1 - d

def att_dice(y_true, y_pred):
    dice = dice_coef(y_true, y_pred).detach()
    loss = (1 - dice) ** 2 *(1 - dice)
    return loss

def masked_dice_loss(y_true, y_pred, mask):
    smooth = 1.
    a = torch.sum(y_true * y_pred * mask, (2, 3, 4))
    b = torch.sum((y_true + y_pred) * mask, (2, 3, 4))
    dice = (2 * a) / (b + smooth)
    return 1 - torch.mean(dice)

def MSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def mix_ce_dice(y_true, y_pred):
    return crossentropy(y_true, y_pred) + 1 - dice_coef(y_true, y_pred)

def crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred+smooth))

def mask_crossentropy(y_pred, y_true, mask):
    smooth = 1e-6
    return -torch.mean(mask * y_true * torch.log(y_pred+smooth))

def B_crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred+smooth)+(1-y_true)*torch.log(1-y_pred+smooth))