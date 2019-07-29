import torch
import torch.nn as nn
import logging


def remove_bn_params(bn_module):
    bn_module.register_buffer('running_mean', None)
    bn_module.register_buffer('running_var', None)
    bn_module.register_parameter('weight', None)
    bn_module.register_parameter('bias', None)


def init_bn_params(bn_module):
    bn_module.running_mean.fill_(0)
    bn_module.running_var.fill_(1)

def absorb_bn(module, bn_module, remove_bn=True, verbose=False):
    with torch.no_grad():
        w = module.weight
        if module.bias is None:
            zeros = torch.zeros(module.out_channels,
                                dtype=w.dtype, device=w.device)
            bias = nn.Parameter(zeros)
            module.register_parameter('bias', bias)
        b = module.bias

        if hasattr(bn_module, 'running_mean'):
            b.add_(-bn_module.running_mean)
        if hasattr(bn_module, 'running_var'):
            invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
            w.mul_(invstd.view(w.size(0), 1, 1, 1))
            b.mul_(invstd)

        if remove_bn:
            if hasattr(bn_module, 'weight'):
                w.mul_(bn_module.weight.view(w.size(0), 1, 1, 1))
                b.mul_(bn_module.weight)
            if hasattr(bn_module, 'bias'):
                b.add_(bn_module.bias)
            remove_bn_params(bn_module)
        else:
            init_bn_params(bn_module)

        if verbose:
            logging.info('BN module %s was asborbed into layer %s' %
                         (bn_module, module))


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)


def search_absorb_bn(model, prev=None, remove_bn=True, verbose=False):
    with torch.no_grad():
        for m in model.children():
            if is_bn(m) and is_absorbing(prev):
                absorb_bn(prev, m, remove_bn=remove_bn, verbose=verbose)
            search_absorb_bn(m, remove_bn=remove_bn, verbose=verbose)
            prev = m
