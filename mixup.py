import torch
import torch.nn as nn
import numpy as np
from numpy.random import beta
from torch.nn.functional import one_hot


class MixUp(nn.Module):
    def __init__(self, alpha=1., batch_dim=0):
        super(MixUp, self).__init__()
        self.batch_dim = batch_dim
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.mix_value = None
        self.mix_index = None

    def mix(self, x1, x2):
        return x2.lerp(x1, self.mix_value)

    def sample(self, batch_size, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        self.mix_index = torch.randperm(batch_size)
        self.mix_value = beta(alpha, alpha)

    def mix_target(self, y, n_class):
        if not self.training or \
                self.mix_value is None:
            return y
        y = one_hot(y, n_class).to(dtype=torch.float)
        idx = self.mix_index.to(device=y.device)
        y_mix = y.index_select(self.batch_dim, idx)
        return self.mix(y, y_mix)

    def forward(self, x):
        if not self.training or \
                self.mix_value is None:
            return x
        idx = self.mix_index.to(device=x.device)
        x_mix = x.index_select(self.batch_dim, idx)
        return self.mix(x, x_mix)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutMix(MixUp):
    def __init__(self, alpha=1., batch_dim=0):
        super(CutMix, self).__init__(alpha, batch_dim)

    def mix_image(self, x1, x2):
        assert not torch.is_tensor(self.mix_value) or \
            self.mix_value.nelement() == 1
        lam = float(self.mix_value)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x1.size(), lam)
        x1[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                   (x1.size()[-1] * x1.size()[-2]))
        self.mix_value = lam
        return x1

    def sample(self, batch_size, alpha=None):
        super(CutMix, self).sample(batch_size, alpha=alpha)

    def forward(self, x):
        if not self.training or \
                self.mix_value is None:
            return x
        idx = self.mix_index.to(device=x.device)
        x_mix = x.index_select(self.batch_dim, idx)
        return self.mix_image(x, x_mix)
