"""BatteryLife task semantics, kept outside the shared training loop."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from data_provider.batterylife import batterylife_trainloader, selected_tasks
from .base import PredictionAdapter


class CellAdapter(PredictionAdapter):
    domain = 'cell'
    default_model = 'tempalte'
    default_tasks = ('soh', 'trajectory', 'rul')
    compatible_models = ('tempalte',)
    config_keys = ('dataset', 'condition', 'context', 'horizon', 'hidden_dim',
                   'early_cycle_threshold', 'charge_discharge_length', 'dataaccess',
                   'eval_cycle_min', 'eval_cycle_max', 'seed', 'split_file', 'sampling', 'split_seed')

    def __init__(self, args):
        self.args = args
        self.tasks = selected_tasks(args)
        self.weights = {task: (0.5 if task == 'rul' and len(self.tasks) > 1 else 1.0)
                        for task in self.tasks}
        self.units = {task: ('BatteryLife record cycles' if task in ('life', 'rul')
                             else 'SOH fraction') for task in self.tasks}

    def loaders(self, state=None):
        scaler = None
        if state:
            record = state['label_scaler']
            scaler = StandardScaler()
            scaler.mean_ = np.asarray(record['mean'])
            scaler.scale_ = np.asarray(record['scale'])
            scaler.var_ = np.asarray(record['var'])
            scaler.n_features_in_ = 1
            scaler.n_samples_seen_ = record['n_samples_seen']
        self.data = batterylife_trainloader(self.args, scaler)
        return self.data

    def state(self):
        train, val, test = self.data
        scaler = train.dataset.label_scaler
        return {
            'protocol': 'BatteryLife',
            'split_source': train.dataset.split_provenance,
            'cycle_coordinate': 'BatteryLife saved-record ordinal (1..N)',
            'life_supervision': 'Published BatteryLife Life labels JSON, unmodified',
            'rul_definition': 'published_life_label - current_record_ordinal',
            'capacity_targets': 'measured_discharge_capacity_in_Ah / nominal_capacity_in_Ah',
            'trajectory_horizon': 'next H saved records',
            'label_scaler': {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                             'var': scaler.var_.tolist(),
                             'n_samples_seen': int(scaler.n_samples_seen_)},
            'splits': {flag: loader.dataset.manifest()
                       for flag, loader in zip(('train', 'val', 'test'), self.data)},
        }

    def forward(self, model, batch, device):
        history, future, life = model(batch['cycle_curves'].to(device))
        outputs = {'soh': history, 'trajectory': future, 'life': life, 'rul': life}
        keys = {'soh': 'history_soh', 'trajectory': 'future_soh',
                'life': 'life_label', 'rul': 'life_label'}
        targets = {task: batch[keys[task]] for task in self.tasks}
        return {task: (outputs[task], targets[task].to(device),
                       torch.ones_like(targets[task], dtype=torch.bool, device=device))
                for task in self.tasks}

    def decode(self, task, prediction, target, batch):
        if task in ('life', 'rul'):
            scaler = self.data[0].dataset.label_scaler
            prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).ravel()
            target = batch['life'].numpy()
            if task == 'rul':
                current = batch['current_cycle'].numpy()
                prediction, target = prediction-current, target-current
        return prediction, target

    def samples(self, batch):
        return [{'sample_id': f'{name}:{int(anchor)}', 'cell_id': name,
                 'current_record': int(anchor), 'raw_cycle_number': float(raw)}
                for name, anchor, raw in zip(batch['cell_id'], batch['current_cycle'],
                                             batch['raw_cycle_number'])]

    def extras(self, decoded, batch):
        if 'rul' in decoded and 'life' not in decoded:
            prediction, target = decoded['rul']
            current = batch['current_cycle'].numpy()
            return {'life': (prediction + current, target + current)}
        return {}
