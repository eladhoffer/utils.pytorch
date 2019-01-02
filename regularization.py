import torch
from .param_filter import FilterParameters
from torch.nn.utils import clip_grad_norm_
import logging


def is_not_bias(name):
    return not name.endswith('bias')


def is_not_bn(module):
    return not isinstance(module, torch.nn.BatchNorm2d)


def sparsity(p):
    return float(p.eq(0).sum()) / p.nelement()


def _norm_exclude_dim(x, dim=0, keepdim=False):
    dims = tuple(set(range(x.dim())) - set([dim]))
    return x.pow(2).sum(dims, keepdim=keepdim).sqrt()


def _renorm(x, dim=0, inplace=False, eps=1e-12):
    if not inplace:
        x = x.clone()
    return x.div_(_norm_exclude_dim(x, dim, keepdim=True))


class Regularizer(object):
    def __init__(self, model, value=1e-3, filter={}, log=False):
        self._model = model
        self._named_parameters = list(
            FilterParameters(model, **filter).named_parameters())
        self.value = value
        self.log = log
        if self.log:
            logging.debug('Applying regularization to parameters: %s',
                          [n for n, _ in self._named_parameters])

    def pre_step(self):
        pass

    def post_step(self):
        pass


class L2Regularization(Regularizer):
    def __init__(self, model, value=1e-3,
                 filter={'parameter_name': is_not_bias,
                         'module': is_not_bn},
                 pre_op=True, post_op=False, **kwargs):
        super(L2Regularization, self).__init__(
            model, value, filter=filter, **kwargs)
        self.pre_op = pre_op
        self.post_op = post_op

    def pre_step(self):
        if self.pre_op:
            with torch.no_grad():
                for _, p in self._named_parameters:
                    p.grad.add_(self.value, p)
            if self.log:
                logging.debug('L2 penalty of %s was applied pre optimization step',
                              self.value)

    def post_step(self):
        if self.post_op:
            with torch.no_grad():
                for _, p in self._named_parameters:
                    p.add_(-self.value, p)
            if self.log:
                logging.debug('L2 penalty of %s was applied post optimization step',
                              self.value)


class WeightDecay(L2Regularization):
    def __init__(self, *kargs, **kwargs):
        super(WeightDecay, self).__init__(*kargs, **kwargs)


class GradClip(Regularizer):
    def __init__(self, *kargs, **kwargs):
        super(GradClip, self).__init__(*kargs, **kwargs)

    def pre_step(self):
        if self.value > 0:
            with torch.no_grad():
                grad = clip_grad_norm_(self._named_parameters, self.value)
            if self.log:
                logging.debug('Gradient value was clipped from %s to %s',
                              grad, self.value)


class L1Regularization(Regularizer):
    def __init__(self, model, value=1e-3,
                 filter={'parameter_name': is_not_bias,
                         'module': is_not_bn},
                 pre_op=False, post_op=True, report_sparsity=False, **kwargs):
        super(L1Regularization, self).__init__(
            model, value, filter=filter, **kwargs)
        self.pre_op = pre_op
        self.post_op = post_op
        self.report_sparsity = report_sparsity

    def pre_step(self):
        if self.pre_op:
            with torch.no_grad():
                for n, p in self._named_parameters:
                    p.grad.add_(self.value, p.sign())
                    if self.report_sparsity:
                        logging.debug('Sparsity for %s is %s', n, sparsity(p))
            if self.log:
                logging.debug('L1 penalty of %s was applied pre optimization step',
                              self.value)

    def post_step(self):
        if self.post_op:
            with torch.no_grad():
                for n, p in self._named_parameters:
                    p.copy_(torch.nn.functional.softshrink(p, self.value))
                    if self.report_sparsity:
                        logging.debug('Sparsity for %s is %s', n, sparsity(p))
            if self.log:
                logging.debug('L1 penalty of %s was applied post optimization step',
                              self.value)


class BoundedWeightNorm(Regularizer):
    def __init__(self, model,
                 filter={'parameter_name': is_not_bias,
                         'module': is_not_bn},
                 dim=0, **kwargs):
        super(BoundedWeightNorm, self).__init__(
            model, 0, filter=filter, **kwargs)
        self.dim = dim

    def pre_step(self):
        self.prev_norms = {}
        with torch.no_grad():
            for n, p in self._named_parameters:
                self.prev_norms[n] = _norm_exclude_dim(p, self.dim, keepdim=True)

    def post_step(self):
        with torch.no_grad():
            for n, p in self._named_parameters:
                _renorm(p, dim=self.dim, inplace=True)
                p.mul_(self.prev_norms[n])
