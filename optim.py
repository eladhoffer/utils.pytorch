import torch
import logging.config
from math import floor
from copy import deepcopy
from six import string_types
from .regime import Regime
from .param_filter import FilterParameters
from . import regularization
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

_OPTIMIZERS = {name: func for name, func in torch.optim.__dict__.items()}
_LRSCHEDULERS = {name: func for name,
                 func in torch.optim.lr_scheduler.__dict__.items()}

try:
    from adabound import AdaBound
    _OPTIMIZERS['AdaBound'] = AdaBound
except ImportError:
    pass


class _EmptySchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(_EmptySchedule, self).__init__(optimizer, last_epoch=-1)
        self.last_epoch = 0

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1


def copy_params(param_target, param_src):
    with torch.no_grad():
        for p_src, p_target in zip(param_src, param_target):
            p_target.copy_(p_src)


def copy_params_grad(param_target, param_src):
    for p_src, p_target in zip(param_src, param_target):
        if p_target.grad is None:
            p_target.backward(p_src.grad.to(dtype=p_target.dtype))
        else:
            p_target.grad.detach().copy_(p_src.grad)


class ModuleFloatShadow(nn.Module):
    def __init__(self, module):
        super(ModuleFloatShadow, self).__init__()
        self.original_module = module
        self.float_module = deepcopy(module)
        self.float_module.to(dtype=torch.float)

    def parameters(self, *kargs, **kwargs):
        return self.float_module.parameters(*kargs, **kwargs)

    def named_parameters(self, *kargs, **kwargs):
        return self.float_module.named_parameters(*kargs, **kwargs)

    def modules(self, *kargs, **kwargs):
        return self.float_module.modules(*kargs, **kwargs)

    def named_modules(self, *kargs, **kwargs):
        return self.float_module.named_modules(*kargs, **kwargs)

    def original_parameters(self, *kargs, **kwargs):
        return self.original_module.parameters(*kargs, **kwargs)

    def original_named_parameters(self, *kargs, **kwargs):
        return self.original_module.named_parameters(*kargs, **kwargs)

    def original_modules(self, *kargs, **kwargs):
        return self.original_module.modules(*kargs, **kwargs)

    def original_named_modules(self, *kargs, **kwargs):
        return self.original_module.named_modules(*kargs, **kwargs)


class OptimRegime(Regime):
    """
    Reconfigures the optimizer according to setting list.
    Exposes optimizer methods - state, step, zero_grad, add_param_group

    Examples for regime:

    1)  "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
          {'epoch': 2, 'optimizer': 'Adam', 'lr': 5e-4},
          {'epoch': 4, 'optimizer': 'Adam', 'lr': 1e-4},
          {'epoch': 8, 'optimizer': 'Adam', 'lr': 5e-5}
         ]"
    2)
        "[{'step_lambda':
            "lambda t: {
            'optimizer': 'Adam',
            'lr': 0.1 * min(t ** -0.5, t * 4000 ** -1.5),
            'betas': (0.9, 0.98), 'eps':1e-9}
         }]"
    """

    def __init__(self, model, regime, defaults={}, filter=None, use_float_copy=False, log=True):
        super(OptimRegime, self).__init__(regime, defaults)
        if filter is not None:
            model = FilterParameters(model, **filter)
        if use_float_copy:
            model = ModuleFloatShadow(model)
            self._original_parameters = list(model.original_parameters())

        self.parameters = list(model.parameters())
        self.optimizer = torch.optim.SGD(self.parameters, lr=0)
        self.regularizer = regularization.Regularizer(model)
        self.use_float_copy = use_float_copy
        self.lr_scheduler = _EmptySchedule(self.optimizer, last_epoch=-1)
        self.schedule_time_frame = 'epoch'
        self.log = log

    def update(self, epoch=None, train_steps=None, metrics=None):
        """adjusts optimizer according to current epoch or steps and training regime.
        """
        updated = False
        if super(OptimRegime, self).update(epoch, train_steps):
            self.adjust(self.setting)
            updated = True
        if self.schedule_time_frame == 'epoch':
            time = int(floor(epoch)) + 1
        elif self.schedule_time_frame == 'step':
            time = train_steps + 1
        else:
            raise ValueError

        if time != self.lr_scheduler.last_epoch:
            prev_lr = self.get_lr()[0]
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metrics, epoch=time)
            self.lr_scheduler.step(epoch=time)
            updated = True
            if prev_lr != self.get_lr()[0] and self.log:
                logging.debug('OPTIMIZER - lr scheduled = %s'
                              % self.get_lr()[0])

        return updated

    def adjust(self, setting):
        """adjusts optimizer according to a setting dict.
        e.g: setting={optimizer': 'Adam', 'lr': 5e-4}
        """
        reset = setting.get('reset', False)
        if 'optimizer' in setting or reset:
            optim_method = _OPTIMIZERS[setting.get('optimizer', 'SGD')]
            if reset:  # reset the optimizer cache:
                self.optimizer = torch.optim.SGD(self.parameters, lr=0)
                if self.log:
                    logging.debug('OPTIMIZER - reset setting')
            if not isinstance(self.optimizer, optim_method):
                self.optimizer = optim_method(self.optimizer.param_groups)
                if self.log:
                    logging.debug('OPTIMIZER - setting method = %s' %
                                  setting['optimizer'])
        for param_group in self.optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    new_val = setting[key]
                    if new_val != param_group[key]:
                        if self.log:
                            logging.debug('OPTIMIZER - setting %s = %s' %
                                          (key, setting[key]))
                        param_group[key] = setting[key]
                        if key == 'lr':
                            param_group['initial_lr'] = param_group['lr']
                            base_lrs = list(map(lambda group: group['lr'],
                                                self.optimizer.param_groups))
                            self.lr_scheduler.base_lrs = base_lrs

                            # fix for AdaBound
                            if hasattr(self.optimizer, 'base_lrs'):
                                self.optimizer.base_lrs = base_lrs

        if 'regularizer' in setting:
            reg_list = deepcopy(setting['regularizer'])
            if not (isinstance(reg_list, list) or isinstance(reg_list, tuple)):
                reg_list = (reg_list,)
            regularizers = []
            for reg in reg_list:
                if isinstance(reg, dict):
                    name = reg.pop('name')
                    regularizers.append((regularization.__dict__[name], reg))
                elif isinstance(reg, regularization.Regularizer):
                    regularizers.append(reg)
                else:  # callable on model
                    regularizers.append(reg(self.regularizer._model))
            self.regularizer = regularization.RegularizerList(self.regularizer._model,
                                                              regularizers)

        if 'lr_scheduler' in setting:
            schedule_config = setting['lr_scheduler']
            if isinstance(schedule_config, _LRScheduler):
                self.lr_scheduler = schedule_config
            elif isinstance(schedule_config, dict):
                name = schedule_config.pop('name')
                self.schedule_time_frame = schedule_config.pop('time_frame',
                                                               'epoch')
                schedule_config['last_epoch'] = self.lr_scheduler.last_epoch
                self.lr_scheduler = _LRSCHEDULERS[name](self.optimizer,
                                                        **schedule_config)
            elif schedule_config is None:
                self.lr_scheduler = _EmptySchedule(self.optimizer,
                                                   last_epoch=self.lr_scheduler.last_epoch)
            else:  # invalid config
                raise NotImplementedError

    def __getstate__(self):
        return {
            'optimizer_state': self.optimizer.__getstate__(),
            'regime': self.regime,
        }

    def __setstate__(self, state):
        self.regime = state.get('regime')
        self.optimizer.__setstate__(state.get('optimizer_state'))

    def state_dict(self):
        """Returns the state of the optimizer as a :class:`dict`.
        """
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        self.optimizer.zero_grad()
        if self.use_float_copy:
            for p in self._original_parameters:
                if p.grad is not None:
                    p.grad.detach().zero_()

    def step(self):
        """Performs a single optimization step (parameter update).
        """
        if self.use_float_copy:
            copy_params_grad(self.parameters, self._original_parameters)
        self.regularizer.pre_step()
        self.optimizer.step()
        self.regularizer.post_step()

        if self.use_float_copy:
            copy_params(self._original_parameters, self.parameters)

    def pre_forward(self):
        """ allows modification pre-forward pass - e.g for regularization
        """
        self.regularizer.pre_forward()

    def pre_backward(self):
        """ allows modification post-forward pass and pre-backward - e.g for regularization
        """
        self.regularizer.pre_backward()

    def get_value(self, key):
        return [group[key] for group in self.optimizer.param_groups]

    def get_lr(self):
        return self.get_value('lr')


class MultiOptimRegime(OptimRegime):

    def __init__(self, *optim_regime_list, log=True):
        self.optim_regime_list = []
        for optim_regime in optim_regime_list:
            assert isinstance(optim_regime, OptimRegime)
            self.optim_regime_list.append(optim_regime)
        self.log = log

    def update(self, epoch=None, train_steps=None):
        """adjusts optimizer according to current epoch or steps and training regime.
        """
        updated = False
        for i, optim in enumerate(self.optim_regime_list):
            current_updated = optim.update(epoch, train_steps)
            if current_updated and self.log:
                logging.debug('OPTIMIZER #%s was updated' % i)
            updated = updated or current_updated
        return updated

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        for optim in self.optim_regime_list:
            optim.zero_grad()

    def step(self):
        """Performs a single optimization step (parameter update).
        """
        for optim in self.optim_regime_list:
            optim.step()

    def pre_forward(self):
        for optim in self.optim_regime_list:
            optim.pre_forward()

    def pre_backward(self):
        for optim in self.optim_regime_list:
            optim.pre_backward()

    def __repr__(self):
        return str([str(optim) for optim in self.optim_regime_list])

    def get_value(self, key):
        return [[group[key] for group in optim.optimizer.param_groups]
                for optim in self.optim_regime_list]

    def get_lr(self):
        return self.get_value('lr')
