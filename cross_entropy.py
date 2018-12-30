import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .misc import onehot


def _is_long(x):
    if hasattr(x, 'data'):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(logits, target, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    onehot_smoothing = False
    ignore_mask = None
    if smooth_eps > 0:
        num_classes = logits.size(-1)
        if _is_long(target):
            if ignore_index >= 0:
                ignore_mask = target.eq(ignore_index)
            target = onehot(target, num_classes).type_as(logits)
        if smooth_dist is None:
            target = (1 - smooth_eps) * target + \
                smooth_eps / num_classes
            onehot_smoothing = True
        else:
            if smooth_dist.dim() < target.dim():
                smooth_dist = smooth_dist.unsqueeze(0)
            target = torch.lerp(
                target, smooth_dist, smooth_eps)

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target):
        return F.cross_entropy(logits, target, weight, ignore_index=ignore_index, reduction=reduction)

    # cross entropy with real target distribution
    lsm = F.log_softmax(logits, dim=-1)

    if weight is not None:
        target = target * weight.unsqueeze(0)
    if ignore_mask is None and weight is None:
        kl = F.kl_div(lsm, target, reduction='sum' if reduction !=
                      'none' else 'none')
    else:
        kl = F.kl_div(lsm, target, reduction='none')
        if ignore_mask is not None:
            kl.masked_fill_(ignore_mask.unsqueeze(-1), 0)
        if weight is not None:
            kl = kl * weight.unsqueeze(0)
        if reduction != 'none':
            kl = kl.sum()

    # for label smoothing with parameter eps:
    if onehot_smoothing:
        entropy = -(math.log(1 - smooth_eps) + smooth_eps *
                    math.log(smooth_eps / ((num_classes - 1) * (1 - smooth_eps))))
    else:
        if ignore_mask is not None:
            target = target.masked_select(ignore_mask.unsqueeze(-1))
        entropy = -(target * target.log()).sum()

    ce = kl + entropy

    if reduction == 'elementwise_mean':
        ce /= logits.size(0)

    return ce


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None):
        super(CrossEntropyLoss, self).__init__(weight=weight,
                                               ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return cross_entropy(input, target, self.weight, self.ignore_index, self.reduction, self.smooth_eps, smooth_dist)

