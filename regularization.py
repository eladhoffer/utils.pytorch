import torch
from .param_filter import FilterParameters, is_not_bn, is_not_bias
from .absorb_bn import search_absorbe_bn
from torch.nn.utils import clip_grad_norm_
import logging


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

    def named_parameters(self):
        for n, p in self._named_parameters:
            yield n, p

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def _pre_parameter_step(self, parameter):
        pass

    def _post_parameter_step(self, parameter):
        pass

    def pre_step(self):
        pass

    def post_step(self):
        pass

    def pre_forward(self):
        pass

    def pre_backward(self):
        pass


class RegularizerList(Regularizer):
    def __init__(self, model, regularization_list):
        """each item must of of format (RegClass, **kwargs) or instance of Regularizer"""
        super(RegularizerList, self).__init__(model)
        self.regularization_list = []
        for regularizer in regularization_list:
            if not isinstance(regularizer, Regularizer):
                reg, reg_params = regularizer
                regularizer = reg(model=model, **reg_params)
            self.regularization_list.append(regularizer)

    def pre_step(self):
        for reg in self.regularization_list:
            reg.pre_step()

    def post_step(self):
        for reg in self.regularization_list:
            reg.post_step()

    def pre_forward(self):
        for reg in self.regularization_list:
            reg.pre_forward()

    def pre_backward(self):
        for reg in self.regularization_list:
            reg.pre_backward()


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
                grad = clip_grad_norm_(self.parameters(), self.value)
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
                self.prev_norms[n] = _norm_exclude_dim(
                    p, self.dim, keepdim=True)
                p.grad.mul_(p/self.prev_norms[n])

    def post_step(self):
        with torch.no_grad():
            for n, p in self._named_parameters:
                _renorm(p, dim=self.dim, inplace=True)
                p.mul_(self.prev_norms[n])


class LARS(Regularizer):
    """Large Batch Training of Convolutional Networks - https://arxiv.org/abs/1708.03888
    """

    def __init__(self, model, value=1, beta=0,
                 filter={'parameter_name': is_not_bias,
                         'module': is_not_bn},
                 **kwargs):
        super(LARS, self).__init__(model, value, filter=filter, **kwargs)
        self.beta = beta

    def pre_step(self):
        with torch.no_grad():
            for _, p in self._named_parameters:
                norm = p.norm()
                grad_norm = p.grad.norm()
                if self.beta > 0:
                    grad_norm += self.beta * norm
                p.grad.mul_(self.value * norm/grad_norm)


class DropConnect(Regularizer):
    def __init__(self, model, value=0,
                 filter={'parameter_name': is_not_bias,
                         'module': is_not_bn},
                 shakeshake=False, **kwargs):
        super(DropConnect, self).__init__(
            model, value=value, filter=filter, **kwargs)
        self.shakeshake = shakeshake

    def _drop_parameters(self):
        self.parameter_copy = {}
        with torch.no_grad():
            for n, p in self._named_parameters:
                self.parameter_copy[n] = p.clone()
                torch.nn.functional.dropout(p, self.value,
                                            training=True, inplace=True)

    def _reassign_parameters(self):
        with torch.no_grad():
            for n, p in self._named_parameters:
                p.copy_(self.parameter_copy.pop(n))

    def pre_forward(self):
        self._drop_parameters()

    def pre_backward(self):
        if self.shakeshake:
            self._reassign_parameters()

    def pre_step(self):
        if not self.shakeshake:
            self._reassign_parameters()


class AbsorbBN(Regularizer):
    def __init__(self, model, remove_bn=False):
        self._model = model
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = 1
        self.remove_bn = remove_bn
        self._removed = False

    def pre_forward(self):
        if self._removed:
            return
        search_absorbe_bn(self._model, remove_bn=self.remove_bn, verbose=False)
        self._removed = self.remove_bn
