from .regime import Regime
import logging.config
from os import path
import torch


class Recorder(Regime):
    def __init__(self, regime, defaults={}):
        self.regime = regime
        self.current_regime_phase = None
        self.setting = defaults
        self.measurments = None

    def get_steps(self):
        return [item['step'] for item in self.regime]

    @staticmethod
    def load(filename, drop_items=[]):
        try:
            measurments = torch.load(
                filename, map_location='cpu')
            for item in drop_items:
                measurments.pop(item)
            return measurments
        except FileNotFoundError:
            return None

    def update(self, train_steps=None, measurments={}):
        """adjusts optimizer according to current epoch or steps and training regime.
        """
        updated = False
        if super(Recorder, self).update(train_steps=train_steps):
            save_file = self.setting.get('save', None)
            if save_file is not None:
                # filename = path.join(self.file_prefix, f'{train_steps}.record')
                torch.save(measurments, save_file)
                logging.debug(f'Saved measurments to {save_file}')
            load_file = self.setting.get('load', None)
            if load_file is not None:
                logging.debug(f'Loaded measurments from {load_file}')
                self.measurments = self.load(load_file,
                                             drop_items=self.setting.get('drop_items', []))

            updated = True
        return updated
