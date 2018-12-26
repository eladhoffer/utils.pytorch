import torch
from misc import filtered_named_parameters


def is_not_bias(name):
    return not name.endswith('bias')


def is_not_bn(module):
    return isinstance(module, torch.nn.BatchNorm2d)


class Regularizer(object):
    def __init__(self, model, value=1e-3, filter_type=None, filter_name=None):
        self._model = model
        if filter_type is None and filter_name is None:
            self._named_parameters = model.named_parameters()
        else:
            self._named_parameters = filtered_named_parameters(model,
                                                               by_type=filter_type,
                                                               by_name=filter_name)
        self.value = value

    def pre_step(self):
        pass

    def post_step(self):
        pass


class L2Regularization(Regularizer):
    def __init__(self, model, value=1e-3,
                 filter_type=is_not_bias,
                 filter_name=is_not_bn,
                 pre_op=True, post_op=False):
        super(L2Regularization, self).__init__(model, value)
        self.pre_op = pre_op
        self.post_op = post_op

    def pre_step(self):
        with torch.no_grad():
            if self.pre_op:
                for _, p in self._named_parameters:
                    p.grad.add_(self.value, p)

    def post_step(self):
        with torch.no_grad():
            if self.pre_op:
                for _, p in self._named_parameters:
                    p.add_(-self.value, p)


class WeightDecay(L2Regularization):
    def __init__(self, *kargs, **kwargs):
        super(WeightDecay, self).__init__(*kargs, **kwargs)
