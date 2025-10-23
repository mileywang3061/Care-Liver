import os
from os.path import join
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from models.UNet import UNet_reg, UNet_seg
from utils.STN import SpatialTransformer, Re_SpatialTransformer
from utils.augmentation import SpatialTransform
from utils.dataloader_care_test_reg import DatasetFromFolder3D as DatasetFromFolder3D_test_reg
from utils.dataloader_care_test_seg import DatasetFromFolder3D as DatasetFromFolder3D_test_seg
from utils.losses import gradient_loss, ncc_loss, MSE, dice_loss, mi_loss
from utils.utils import AverageMeter
import evaluation as E
import nibabel as nib
from utils.dataloader_care_test_seg import apply_affine

class BRBS(object):
    def __init__(self, k=0,
                 n_channels=1,
                 n_classes=8,
                 lr=1e-4,
                 epoches=200,
                 iters=200,
                 checkpoint_dir='',
                 result_dir='',
                 model_name=''):
        super(BRBS, self).__init__()
        # initialize parameters
        self.k = k
        self.n_classes = n_classes
        self.epoches = epoches
        self.iters = iters
        self.lr = lr

        self.results_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        # tools
        self.stn = SpatialTransformer() # Spatial Transformer
        self.rstn = Re_SpatialTransformer() # Spatial Transformer-inverse
        self.softmax = nn.Softmax(dim=1)

        # data augmentation
        self.spatial_aug = SpatialTransform(do_rotation=True,
                                            angle_x=(-np.pi / 9, np.pi / 9),
                                            angle_y=(-np.pi / 9, np.pi / 9),
                                            angle_z=(-np.pi / 9, np.pi / 9),
                                            do_scale=True,
                                            scale=(0.75, 1.25))

        # initialize networks
        self.Reger = UNet_reg(n_channels=n_channels)
        self.Seger = UNet_seg(n_channels=n_channels, n_classes=n_classes)

        if torch.cuda.is_available():
            self.Reger = self.Reger.cuda()
            self.Seger = self.Seger.cuda()

        # initialize optimizer
        self.optR = torch.optim.Adam(self.Reger.parameters(), lr=lr)
        self.optS = torch.optim.Adam(self.Seger.parameters(), lr=lr)

    def test_iterator_seg(self, mi):
        with torch.no_grad():
            # Seg
            s_m = self.Seger(mi)
        return s_m

    def test_iterator_reg(self, mi, fi, ml=None, fl=None):
        with torch.no_grad():
            # Reg
            w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow = self.Reger(mi, fi, ml, fl)

        return w_m_to_f, w_label_m_to_f, flow

    def test(self, test_dir, batch_size=1):
        test_dataset_seg = DatasetFromFolder3D_test_seg(test_dir, self.n_classes)
        self.dataloader_test_seg = DataLoader(test_dataset_seg, batch_size=batch_size, shuffle=False)
        # test_dataset_reg = DatasetFromFolder3D_test_reg(test_dir, self.n_classes)
        # self.dataloader_test_reg = DataLoader(test_dataset_reg, batch_size=batch_size)

        self.Seger.eval()
        # self.Reger.eval()
        dice_seg = []
        for i, (mi, ml, name, affine) in enumerate(self.dataloader_test_seg):
            # name = os.path.basename(os.path.dirname(name[0]))
            name = os.path.basename(name[0]).split('.nii.gz')[0]
            if torch.cuda.is_available():
                mi = mi.cuda()
            s_m = self.test_iterator_seg(mi)
            s_m_ = np.argmax(s_m.data.cpu().numpy()[0], axis=0)
            s_m = s_m_.astype(np.uint8)
            if not os.path.exists(join(self.results_dir, self.model_name, 'seg')):
                os.makedirs(join(self.results_dir, self.model_name, 'seg'))

            # s_m = sitk.GetImageFromArray(s_m)
            # sitk.WriteImage(s_m, join(self.results_dir, self.model_name, 'seg', name+'.nii'))
            s_m = apply_affine(s_m.transpose([2,1,0]), affine[0])
            nib.save(nib.Nifti1Image(s_m, affine[0]), join(self.results_dir, self.model_name, 'seg', name+'_mask.nii.gz'))
            dice = [0]
            if ml.sum() != 0:
                onehot = E.OneHot([0,1])
                dice = E.dice_coefficient(ml[0].numpy(), onehot(s_m_), axis=(1,2,3))[1:]
                dice_seg.append(dice)
            print(name+'.nii: ', f'dice: {np.mean(dice)}')

        # sio.savemat(join(self.results_dir, self.model_name, 'dice_seg.mat'), {'dice_seg': dice_seg})

        # dice_reg_baseline = []
        # dice_reg = []
        # for i, (mi, fi, name1, name2, affine1, affine2) in enumerate(self.dataloader_test_reg):
        #     # if i == 725:
        #     #     break
        #     # name1 = os.path.basename(os.path.dirname(name1[0]))
        #     # name2 = os.path.basename(os.path.dirname(name2[0]))
        #     name1 = os.path.basename(name1[0]).split('.nii.gz')[0]
        #     name2 = os.path.basename(name2[0]).split('.nii.gz')[0]
        #     if name1 is not name2:
        #         if torch.cuda.is_available():
        #             mi = mi.cuda()
        #             fi = fi.cuda()
        #             # ml = ml.cuda()
        #             # fl = fl.cuda()
        #         ml = None
        #         fl = None

        #         w_m_to_f, w_label_m_to_f, flow = self.test_iterator_reg(mi, fi, ml, fl)

        #         flow = flow.data.cpu().numpy()[0]
        #         w_m_to_f = w_m_to_f.data.cpu().numpy()[0, 0]
                
        #         flow = flow.astype(np.float32)
        #         w_m_to_f = w_m_to_f.astype(np.float32)

        #         if ml is not None and fl is not None:
        #             w_label_m_to_f_ = np.argmax(w_label_m_to_f.data.cpu().numpy()[0], axis=0)
        #             w_label_m_to_f = w_label_m_to_f_.astype(np.uint8)

        #         if not os.path.exists(join(self.results_dir, self.model_name, 'flow')):
        #             os.makedirs(join(self.results_dir, self.model_name, 'flow'))
        #         if not os.path.exists(join(self.results_dir, self.model_name, 'w_m_to_f')):
        #             os.makedirs(join(self.results_dir, self.model_name, 'w_m_to_f'))
        #         if ml is not None and fl is not None and not os.path.exists(join(self.results_dir, self.model_name, 'w_label_m_to_f')):
        #             os.makedirs(join(self.results_dir, self.model_name, 'w_label_m_to_f'))

        #         # w_m_to_f = sitk.GetImageFromArray(w_m_to_f)
                
        #         # sitk.WriteImage(w_m_to_f, join(self.results_dir, self.model_name, 'w_m_to_f', name2+'_'+name1+'.nii'))
        #         # w_label_m_to_f = sitk.GetImageFromArray(w_label_m_to_f)
        #         # sitk.WriteImage(w_label_m_to_f, join(self.results_dir, self.model_name, 'w_label_m_to_f', name2+'_'+name1+'.nii'))
        #         # flow = sitk.GetImageFromArray(flow)
        #         # sitk.WriteImage(flow, join(self.results_dir, self.model_name, 'flow', name2+'_'+name1+'.nii'))

        #         w_m_to_f = apply_affine(w_m_to_f.transpose([2,1,0]), affine2[0])
        #         flow = apply_affine(flow.transpose([3,2,1,0]), affine2[0])
        #         nib.save(nib.Nifti1Image(w_m_to_f, affine2[0]), join(self.results_dir, self.model_name, 'w_m_to_f', name1+'_regto_'+name2+'.nii.gz'))
        #         nib.save(nib.Nifti1Image(flow, affine2[0]), join(self.results_dir, self.model_name, 'flow', name1+'_regto_'+name2+'_disp.nii.gz'))
        #         if ml is not None and fl is not None:
        #             w_label_m_to_f = apply_affine(w_label_m_to_f.transpose([2,1,0]), affine2[0])
        #             nib.save(nib.Nifti1Image(w_label_m_to_f, affine2[0]), join(self.results_dir, self.model_name, 'w_label_m_to_f', name1+'_regto_'+name2+'_mask.nii.gz'))


        #         if ml is not None and fl is not None and name1 != name2:
        #             onehot = E.OneHot([0,1])
        #             dice_b = E.dice_coefficient(fl[0].cpu().numpy(), ml[0].cpu().numpy(), axis=(1,2,3))[1:]
        #             dice_reg_baseline.append(dice_b)
        #             dice = E.dice_coefficient(fl[0].cpu().numpy(), onehot(w_label_m_to_f_), axis=(1,2,3))[1:]
        #             dice_reg.append(dice)
        #             print(name1+'_regto_'+name2+'.nii' + f' - baseline: {np.mean(dice_b)}, dice: {np.mean(dice)}')

        #         else:
        #             print(name1+'_regto_'+name2+'.nii')

        #     sio.savemat(join(self.results_dir, self.model_name, 'dice_reg.mat'), {'dice_reg_baseline': dice_reg_baseline, 'dice_reg': dice_reg})

    def checkpoint(self, epoch, k):
        torch.save(self.Seger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Seger_'+self.model_name, epoch+k),
                   _use_new_zipfile_serialization=False)
        torch.save(self.Reger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Reger_'+self.model_name, epoch+k),
                   _use_new_zipfile_serialization=False)

    def load(self, k=0):
        # self.Reger.load_state_dict(
        #     torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Reger_'+self.model_name, str(k))))
        self.Seger.load_state_dict(
            torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Seger_' + self.model_name, str(k))))


def run_test(processed_dir='./processed/'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    result_dir = 'results_care_256_256_32_mi'
    ckpt_dir = f'{result_dir}/weights'
    model_name = 'BRBS_care_all'
    test_dir = processed_dir

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    RSTNet = BRBS(n_classes=2,epoches=80, iters=1000,
                 checkpoint_dir=ckpt_dir,
                 result_dir=result_dir,
                 model_name=model_name)
    RSTNet.load(k=80)
    RSTNet.test(test_dir=test_dir)



