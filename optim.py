import torch
import logging.config
from copy import deepcopy
from six import string_types
from regime import Regime
from misc import filtered_named_parameters
import regularization


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

    def __init__(self, model, regime, defaults={}, filter_type=None, filter_name=None):
        super(OptimRegime, self).__init__(regime, defaults)
        if filter_type is None and filter_name is None:
            self._named_parameters = model.named_parameters()
        else:
            self._named_parameters = filtered_named_parameters(model,
                                                               by_type=filter_type,
                                                               by_name=filter_name)
        params = (param for _, param in self._named_parameters)
        self.optimizer = torch.optim.SGD(params, lr=0)
        self.regularizer = regularization.Regularizer(params)

    def update(self, epoch=None, train_steps=None):
        """adjusts optimizer according to current epoch or steps and training regime.
        """
        if super(OptimRegime, self).update(epoch, train_steps):
            self.adjust(self.setting)
            return True
        else:
            return False

    def adjust(self, setting):
        """adjusts optimizer according to a setting dict.
        e.g: setting={optimizer': 'Adam', 'lr': 5e-4}
        """
        if 'optimizer' in setting:
            optim_method = torch.optim.__dict__[setting['optimizer']]
            if not isinstance(self.optimizer, optim_method):
                self.optimizer = optim_method(self.optimizer.param_groups)
                logging.debug('OPTIMIZER - setting method = %s' %
                              setting['optimizer'])
        for param_group in self.optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    new_val = setting[key]
                    if new_val != param_group[key]:
                        logging.debug('OPTIMIZER - setting %s = %s' %
                                      (key, setting[key]))
                        param_group[key] = setting[key]

        if 'regularizer' in setting:
            reg_setting = deepcopy(setting['regularizer'])
            name = reg_setting.pop('name')
            self.regularizer = regularization.__dict__[name](self.regularizer.parameters,
                                                             **reg_setting)

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
        return {
            'optimizer_state': self.optimizer.state_dict(),
            'regime': self.regime,
        }

    def load_state_dict(self, state_dict):
        """Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        optimizer_state_dict = state_dict['optimizer_state']

        self.__setstate__({'optimizer_state': optimizer_state_dict,
                           'regime': state_dict['regime']})

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        self.optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.regularizer.pre_step()
        self.optimizer.step(closure)
        self.regularizer.post_step()


class MultiOptimRegime(Regime):

    def __init__(self, model, regime, defaults={}, filter_type=None, filter_name=None):
        super(OptimRegime, self).__init__(regime, defaults)
        if filter_type is None and filter_name is None:
            params = model.parameters()
        else:
            filtered = filtered_named_parameters(model,
                                                 by_type=filter_type,
                                                 by_name=filter_name)
            params = (param for _, param in filtered)
        self.optimizer = torch.optim.SGD(params, lr=0)

    def update(self, epoch=None, train_steps=None):
        """adjusts optimizer according to current epoch or steps and training regime.
        """
        if super(OptimRegime, self).update(epoch, train_steps):
            self.adjust(self.setting)
            return True
        else:
            return False

    def adjust(self, setting):
        """adjusts optimizer according to a setting dict.
        e.g: setting={optimizer': 'Adam', 'lr': 5e-4}
        """
        if 'optimizer' in setting:
            optim_method = torch.optim.__dict__[setting['optimizer']]
            if not isinstance(self.optimizer, optim_method):
                self.optimizer = optim_method(self.optimizer.param_groups)
                logging.debug('OPTIMIZER - setting method = %s' %
                              setting['optimizer'])
        for param_group in self.optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    new_val = setting[key]
                    if new_val != param_group[key]:
                        logging.debug('OPTIMIZER - setting %s = %s' %
                                      (key, setting[key]))
                        param_group[key] = setting[key]

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
        return {
            'optimizer_state': self.optimizer.state_dict(),
            'regime': self.regime,
        }

    def load_state_dict(self, state_dict):
        """Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        optimizer_state_dict = state_dict['optimizer_state']

        self.__setstate__({'optimizer_state': optimizer_state_dict,
                           'regime': state_dict['regime']})

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        self.optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.optimizer.step(closure)
