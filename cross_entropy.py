import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .misc import onehot


def _is_long(x):
    if hasattr(x, 'data'):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(logits, target, weight=None, size_average=True,
                  ignore_index=-100, reduce=True, smooth_eps=None, smooth_dist=None):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    onehot_smoothing = False
    if smooth_eps > 0:
        num_classes = logits.size(-1)
        mask_idx = None
        if _is_long(target):
            if ignore_index >= 0:
                mask_idx = target.eq(ignore_index)
            target = onehot(target, num_classes).type_as(logits)
        if smooth_dist is None:
            target = (1 - smooth_eps) * target + \
                smooth_eps / num_classes
            onehot_smoothing = True
        else:
            target = torch.lerp(
                target, smooth_dist.unsqueeze(0), smooth_eps)

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target):
        return F.cross_entropy(logits, target, weight, size_average,
                               ignore_index, reduce)

    # cross entropy with real target distribution
    lsm = F.log_softmax(logits, dim=-1)

    if weight is not None:
        target = target * weight.unsqueeze(0)
    if mask_idx is None and weight is None:
        kl = F.kl_div(lsm, target, size_average=size_average, reduce=reduce)
    else:
        kl = F.kl_div(lsm, target, size_average=size_average, reduce=False)
        if mask_idx is not None:
            kl.masked_fill_(mask_idx.unsqueeze(1), 0)
        if weight is not None:
            kl = kl * weight.unsqueeze(0)
        if reduce:
            kl = kl.mean() if size_average else kl.sum()

    # for label smoothing with parameter eps:
    if onehot_smoothing:
        entropy = -(math.log(1 - smooth_eps) + smooth_eps *
                    math.log(smooth_eps / ((num_classes - 1) * (1 - smooth_eps))))
    else:
        entropy = -(target * target.log()).sum()

    if size_average:
        kl *= num_classes
        entropy /= logits.size(0)
    ce = kl + entropy

    return ce


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True,
                 smooth_eps=None, smooth_dist=None):
        super(CrossEntropyLoss, self).__init__(
            weight, size_average, ignore_index, reduce)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist

    def forward(self, input, target):
        return cross_entropy(input, target, self.weight, self.size_average,
                             self.ignore_index, self.reduce, self.smooth_eps, self.smooth_dist)
