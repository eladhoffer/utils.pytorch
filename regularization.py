import torch
from .param_filter import FilterParameters, is_not_bn, is_not_bias
from .absorb_bn import search_absorb_bn
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


def _norm(x, dim, p=2):
    """Computes the norm over all dimensions except dim"""
    if p == -1:
        def func(x, dim): return x.max(dim=dim)[0] - x.min(dim=dim)[0]
    elif p == float('inf'):
        def func(x, dim): return x.max(dim=dim)[0]
    else:
        def func(x, dim): return torch.norm(x, dim=dim, p=p)
    if dim is None:
        return x.norm(p=p)
    elif dim == 0:
        output_size = (x.size(0),) + (1,) * (x.dim() - 1)
        return func(x.contiguous().view(x.size(0), -1), 1).view(*output_size)
    elif dim == x.dim() - 1:
        output_size = (1,) * (x.dim() - 1) + (x.size(-1),)
        return func(x.contiguous().view(-1, x.size(-1)), 0).view(*output_size)
    else:
        return _norm(x.transpose(0, dim), 0).transpose(0, dim)


def _param_grad_norm(parameters):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


class Regularizer(object):
    def __init__(self, model, value=0, filter={}, log=False):
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
    def __init__(self, model, value=0,
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
                    p.grad.add_(p, alpha=self.value)
            if self.log:
                logging.debug('L2 penalty of %s was applied pre optimization step',
                              self.value)

    def post_step(self):
        if self.post_op:
            with torch.no_grad():
                for _, p in self._named_parameters:
                    p.add_(p, alpha=-self.value)
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
                logging.debug('Gradient norm value was clipped from %s to %s',
                              grad, self.value)


class AngleNoise(Regularizer):
    def __init__(self, *kargs, **kwargs):
        super(AngleNoise, self).__init__(*kargs, **kwargs)

    def pre_step(self):
        if self.value > 0:
            with torch.no_grad():
                parameters = list(
                    filter(lambda p: p.grad is not None, self.parameters()))
                total_norm = _param_grad_norm(parameters)
                std = self.value
                for p in parameters:
                    grad = p.grad
                    grad.div_(total_norm)
                    noise = torch.empty_like(grad).normal_(std=std)
                    grad.add_(noise)
                new_norm = _param_grad_norm(parameters)
                scale = total_norm / new_norm
                for p in parameters:
                    p.grad.mul_(scale)
                # num_params = sum([p.numel() for p in parameters])
                # std = self.value
                # noise_norm = std * (num_params ** 0.5)
                # for p in parameters:
                #     grad = p.grad
                #     grad.div_(total_norm + noise_norm)
                #     noise = torch.empty_like(grad).normal_(std=std)
                #     grad.add_(noise).mul_(total_norm)

            if self.log:
                logging.debug('Gradient angle was added noise with std value %s',
                              self.value)


class GradSmooth(Regularizer):
    def __init__(self,  model, value=True, momentum=0.9, filter={}, log=False):
        super(GradSmooth, self).__init__(model,
                                         value=value, filter=filter, log=log)
        self.momentum = momentum
        self.running_norm = None
        self.enabled = value
        self.counter = 0

    def pre_step(self):
        parameters = list(
            filter(lambda p: p.grad is not None, self.parameters()))
        total_norm = _param_grad_norm(parameters)
        if self.running_norm is None:
            self.running_norm = total_norm
        else:
            self.running_norm = self.momentum * self.running_norm \
                + (1-self.momentum) * total_norm
            if self.enabled:
                clip_coef = self.running_norm / (total_norm + 1e-6)
                for p in parameters:
                    p.grad.data.mul_(clip_coef)
            if self.log:
                logging.debug('Gradient norm value was clipped from %s to %s',
                              total_norm, self.running_norm)

        self.counter += 1


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
                    p.grad.add_(p.sign(), alpha=self.value)
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
                 dim=0, p=2, **kwargs):
        super(BoundedWeightNorm, self).__init__(
            model, 0, filter=filter, **kwargs)
        self.dim = dim
        self.init_norms = None
        self.p = p

    def _gather_init_norm(self):
        self.init_norms = {}
        with torch.no_grad():
            for n, p in self._named_parameters:
                self.init_norms[n] = _norm(
                    p, self.dim, p=self.p).detach().mean()

    def pre_forward(self):
        if self.init_norms is None:
            self._gather_init_norm()
        with torch.no_grad():
            for n, p in self._named_parameters:
                init_norm = self.init_norms[n]
                new_norm = _norm(p, self.dim, p=self.p)
                p.mul_(init_norm / new_norm)

    def pre_step(self):
        for n, p in self._named_parameters:
            init_norm = self.init_norms[n]
            norm = _norm(p, self.dim, p=self.p)
            curr_grad = p.grad.data.clone()
            p.grad.data.zero_()
            p_normed = p * (init_norm / norm)
            p_normed.backward(curr_grad)


class LARS(Regularizer):
    """Large Batch Training of Convolutional Networks - https://arxiv.org/abs/1708.03888
    """

    def __init__(self, model, value=0.01, weight_decay=0, dim=None, p=2, min_scale=None, max_scale=None,
                 filter={'parameter_name': is_not_bias,
                         'module': is_not_bn},
                 **kwargs):
        super(LARS, self).__init__(model, value, filter=filter, **kwargs)
        self.weight_decay = weight_decay
        self.dim = dim
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale

    def pre_step(self):
        with torch.no_grad():
            for _, param in self._named_parameters:
                param.grad.add_(param, alpha=self.weight_decay)
                if self.dim is not None:
                    norm = _norm(param, dim=self.dim, p=self.p)
                    grad_norm = _norm(param.grad, dim=self.dim, p=self.p)
                else:
                    norm = param.norm(p=self.p)
                    grad_norm = param.grad.norm(p=self.p)
                scale = self.value * norm/grad_norm
                if self.min_scale is not None or self.max_scale is not None:
                    scale.clamp_(min=self.min_scale, max=self.max_scale)
                param.grad.mul_(scale)


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
        if not remove_bn:
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.momentum = 1
        self.remove_bn = remove_bn
        self._removed = False

    def pre_forward(self):
        if self._removed:
            return
        search_absorb_bn(self._model, remove_bn=self.remove_bn, verbose=False)
        self._removed = self.remove_bn


class Consolidate(Regularizer):
    """ groups is a list of tuples, each tuple is of consolidated parameters (of same size)
    """

    def __init__(self, model, value=0.1, force=False, **kwargs):
        super(Consolidate, self).__init__(
            model, value, **kwargs)
        self.force = force
        shared_params = {}
        for name, param in self._named_parameters:
            numel = param.numel()
            shared_params.setdefault(numel, [])
            shared_params[numel].append((name, param))

        self.groups = []
        shared_names = []
        for shared_list in shared_params.values():
            if len(shared_list) < 2:
                continue
            names, params = zip(*shared_list)
            self.groups.append(params)
            shared_names.append(names)

        logging.debug('Shared parameter groups %s' % shared_names)

    def pre_step(self):
        with torch.no_grad():
            for i, group in enumerate(self.groups):
                mean_group = sum(group) / len(group)
                for p in group:
                    p.grad.data.add_(p - mean_group, alpha=self.value)
                if self.log:
                    logging.debug('group %s, diff norm = %s' %
                                  (i, float(sum([(p - mean_group).norm() ** 2 for p in group]).sqrt())))

    def post_step(self):
        if self.force:
            with torch.no_grad():
                for group in enumerate(self.groups):
                    mean_group = sum(group) / len(group)
                    for p in group:
                        p.copy_(mean_group)
