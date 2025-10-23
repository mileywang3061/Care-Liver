
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Patch Generation for Liver Fibrosis Challenge")
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (1, 2, 3, or 4)')
    parser.add_argument('--patch_size', type=int, default=16, help='Size of the patches to be extracted')
    parser.add_argument('--cover_rate', type=float, default=1.0, help='Coverage rate for patch extraction')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training ')
    parser.add_argument('--data_path',type=str, help='Root path to MRI folders'),
    parser.add_argument('--fold_path', type=str, help='Path to save the generated patches'),
    parser.add_argument('--mask_path', type=str, help='Path to the mask files'),
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_step_size', type=int, default=10, help='Step size for LR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.7, help='Gamma for LR scheduler')   
    parser.add_argument('--data_type', type=str, default='non_contrast', help='contrast or non_contrast')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    return parser.parse_args()