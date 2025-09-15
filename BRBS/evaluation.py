import numpy as np
def dice_coefficient(pred, gt, epsilon=1e-9, ignore_background=False, axis=None):
    """ 2 * intersection(pred, gt) / (pred + gt) 
        2 * tp / (2*tp + fp + fn)
    """
    axis = tuple(range(2, pred.ndim)) if axis is None else axis
    intersection = (pred * gt).sum(axis)
    sum_ = (pred + gt).sum(axis)
    dice = 2 * intersection / (sum_ + epsilon)
    if ignore_background:
        dice = dice[:, 1:]
    return dice

class OneHot:
    def __init__(self, n_classes, dtype=np.float32):
        self.n_classes = n_classes
        self.dtype = dtype

    def __call__(self, x):
        nc = list(range(self.n_classes)) if type(self.n_classes) is int else self.n_classes
        if type(nc) in [list, tuple, np.ndarray]:
            x_ = np.zeros([len(nc)] + list(x.shape), self.dtype)
            for i in range(len(nc)):
                x_[i, ...][x == nc[i]] = 1
        else:
            raise Exception('Wrong type of n_class.')
        return x_    
    