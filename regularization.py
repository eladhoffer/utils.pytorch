import torch
from param_filter import FilterParameters


def is_not_bias(name):
    return not name.endswith('bias')


def is_not_bn(module):
    return isinstance(module, torch.nn.BatchNorm2d)


class Regularizer(object):
    def __init__(self, model, value=1e-3, filter={}):
        self._model = model
        self._parameters = list(FilterParameters(model, **filter).parameters())
        self.value = value

    def pre_step(self):
        pass

    def post_step(self):
        pass


class L2Regularization(Regularizer):
    def __init__(self, model, value=1e-3,
                 filter={'parameter_name': is_not_bias,
                         'module_type': is_not_bn},
                 pre_op=True, post_op=False):
        super(L2Regularization, self).__init__(model, value, filter=filter)
        self.pre_op = pre_op
        self.post_op = post_op

    def pre_step(self):
        with torch.no_grad():
            if self.pre_op:
                for p in self._parameters:
                    p.grad.add_(self.value, p)

    def post_step(self):
        with torch.no_grad():
            if self.pre_op:
                for p in self._parameters:
                    p.add_(-self.value, p)


class WeightDecay(L2Regularization):
    def __init__(self, *kargs, **kwargs):
        super(WeightDecay, self).__init__(*kargs, **kwargs)
