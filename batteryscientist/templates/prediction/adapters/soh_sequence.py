"""Capacity-history forecasting is a separate input protocol from IC2ML curves."""
import torch
from .base import PredictionAdapter
from data_provider.soh_sequence import sequence_loaders


class SOHSequenceAdapter(PredictionAdapter):
    domain = 'cell'
    default_model = 'SOHLinear'
    default_tasks = ('trajectory',)
    compatible_models = ('SOHPersistence', 'SOHTrend', 'SOHLinear', 'SOHMLP', 'SOHTCN')
    config_keys = ('dataset', 'context', 'horizon', 'split_file', 'cycle_coordinate', 'cycle_step',
                   'eval_cycle_min', 'eval_cycle_max', 'sampling', 'dataaccess', 'split_seed')

    def __init__(self, args):
        self.args = args
        self.tasks = tuple(args.tasks)
        if self.tasks != ('trajectory',):
            raise ValueError('SOH sequence adapter supports tasks=[trajectory]')
        self.weights = {'trajectory': 1.0}
        self.units = {'trajectory': 'SOH fraction'}

    def loaders(self, state=None):
        self.data, self.preprocessing = sequence_loaders(self.args, state)
        return self.data

    def state(self):
        return self.preprocessing

    def forward(self, model, batch, device):
        prediction = model(batch['history'].to(device))
        target = batch['target'].to(device)
        return {'trajectory': (prediction, target, torch.ones_like(target, dtype=torch.bool))}

    def decode(self, task, prediction, target, batch):
        return prediction, target

    def samples(self, batch):
        return [dict(sample_id=sample_id, cell_id=cell_id, current_record=int(anchor),
                     raw_cycle_number=float(raw))
                for sample_id, cell_id, anchor, raw in zip(batch['sample_id'], batch['cell_id'],
                                                          batch['current_record'], batch['raw_cycle_number'])]
