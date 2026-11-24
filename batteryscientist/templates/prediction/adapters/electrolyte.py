"""Electrolyte property regression with measured-label masks."""

from data_provider.electrolyte import electrolyte_loaders
from .base import PredictionAdapter


class ElectrolyteAdapter(PredictionAdapter):
    domain = 'electrolyte'
    default_model = 'CompositionMLP'
    default_tasks = ('conductivity',)
    compatible_models = ('CompositionMLP',)
    config_keys = ('dataset', 'hidden_dim', 'fingerprint_dim', 'split_seed',
                   'val_fraction', 'test_fraction', 'split_file')

    def __init__(self, args):
        self.args = args
        self.tasks = tuple(dict.fromkeys(args.tasks))
        allowed = {'conductivity', 'anion_ratio'}
        if not self.tasks or set(self.tasks) - allowed:
            raise ValueError(f'Electrolyte tasks must be a nonempty selection of {sorted(allowed)}')
        if args.dataset != 'bamboo_mixer':
            raise ValueError('Electrolyte adapter currently supports dataset=bamboo_mixer')
        self.weights = dict.fromkeys(self.tasks, 1.0)
        # Preserve source units rather than infer units from property names.
        self.units = {task: ('fraction' if task == 'anion_ratio' else 'source dataset units')
                      for task in self.tasks}

    def loaders(self, state=None):
        self.data, self.preprocessing = electrolyte_loaders(self.args, state)
        return self.data

    def state(self):
        return self.preprocessing

    def forward(self, model, batch, device):
        outputs = model(batch['features'].to(device))
        return {task: (outputs[task], batch['targets'][task].to(device),
                       batch['target_masks'][task].to(device)) for task in self.tasks}

    def decode(self, task, prediction, target, batch):
        record = self.preprocessing['target_scalers'][task]
        return prediction * record['scale'] + record['mean'], target * record['scale'] + record['mean']

    def samples(self, batch):
        return [{'sample_id': sample_id, 'formulation_group': group}
                for sample_id, group in zip(batch['sample_id'], batch['group_id'])]

