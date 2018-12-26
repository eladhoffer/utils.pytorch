import torch


class Regularizer(object):
    def __init__(self, parameters, value=1e-3):
        self.parameters = parameters
        self.value = value

    def pre_step(self):
        pass

    def post_step(self):
        pass


class L2Regularization(Regularizer):
    def __init__(self, parameters, value=1e-3, pre_op=True, post_op=False):
        super(L2Regularization, self).__init__(parameters, value)
        self.pre_op = pre_op
        self.post_op = post_op

    def pre_step(self):
        with torch.no_grad():
            if self.pre_op:
                for p in self.parameters:
                    p.grad.add_(self.value, p)

    def post_step(self):
        with torch.no_grad():
            if self.pre_op:
                for p in self.parameters:
                    p.add_(-self.value, p)


class WeightDecay(L2Regularization):
    def __init__(self, parameters, value=1e-3, pre_op=True, post_op=False):
        super(WeightDecay, self).__init__(parameters,
                                          value=value, pre_op=pre_op, post_op=post_op)
