"""Bamboo-mixer JSON adapter with formulation groups and train-only scaling.

Inputs use deterministic SMILES character n-grams, weighted by component molar
fractions, plus salt fraction and temperature. This is a lightweight baseline
representation, not a chemical graph fingerprint or a pretrained Bamboo encoder.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

FLAGS = ('train', 'val', 'test')


def components(items):
    combined = {}
    if not isinstance(items, list) or not items:
        raise ValueError('Each formulation needs nonempty solvents and salts lists')
    for item in items:
        smiles = item['smiles']
        ratio = float(item['molar_ratio'])
        if not isinstance(smiles, str) or not smiles.strip() or not np.isfinite(ratio) or ratio < 0:
            raise ValueError('Invalid component SMILES or molar_ratio')
        combined[smiles] = combined.get(smiles, 0.0) + ratio
    total = sum(combined.values())
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Component molar ratios must have a positive sum')
    return [(smiles, ratio / total) for smiles, ratio in sorted(combined.items()) if ratio > 0]


def formulation_key(row):
    """Ignore temperature so repeated measurements stay in the same partition.

    Grouping uses source SMILES strings and normalized molar ratios; it does not
    claim to identify chemically equivalent alternative SMILES representations.
    """
    salt_fraction = float(row['salt_molar_ratio'])
    temperature = float(row['temperature'])
    if not np.isfinite(temperature) or not np.isfinite(salt_fraction) or not 0 <= salt_fraction <= 1:
        raise ValueError('Invalid temperature or salt_molar_ratio')
    value = {kind: [(smiles, round(ratio, 10)) for smiles, ratio in components(row[kind])]
             for kind in ('solvents', 'salts')}
    value['salt_molar_ratio'] = round(salt_fraction, 10)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def features(row, dimension):
    result = np.zeros(2 * dimension + 2, dtype=np.float64)
    for channel, kind in enumerate(('solvents', 'salts')):
        for smiles, ratio in components(row[kind]):
            tokens = [smiles[i:i+width] for width in (1, 2, 3)
                      for i in range(len(smiles)-width+1)]
            for token in tokens:
                bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'big') % dimension
                result[channel * dimension + bucket] += ratio / len(tokens)
    result[-2:] = [row['salt_molar_ratio'], row['temperature']]
    return result


def target(row, task):
    mask = row.get(f'{task}_mask', False)
    if not isinstance(mask, bool):
        raise ValueError(f'{task}_mask must be a JSON boolean')
    if not mask:
        return 0.0, False
    value = float(row[task])
    if not np.isfinite(value):
        raise ValueError(f'Non-finite measured label for {task}')
    return value, True


def partition(groups, args):
    split_file = getattr(args, 'split_file', None)
    if split_file:
        result = json.loads(Path(split_file).read_text())
    else:
        unique = sorted(set(groups))
        if len(unique) < 3:
            raise ValueError('At least three distinct formulation groups are required')
        np.random.default_rng(args.split_seed).shuffle(unique)
        n_val = max(1, round(len(unique) * args.val_fraction))
        n_test = max(1, round(len(unique) * args.test_fraction))
        if n_val + n_test >= len(unique):
            raise ValueError('Split fractions leave no training formulation groups')
        lookup = {group: ('val' if i < n_val else 'test' if i < n_val+n_test else 'train')
                  for i, group in enumerate(unique)}
        result = {flag: [i for i, group in enumerate(groups) if lookup[group] == flag] for flag in FLAGS}
    validate_partition(result, groups)
    return result


def validate_partition(result, groups):
    if set(result) != set(FLAGS) or any(not result[flag] for flag in FLAGS):
        raise ValueError('Split must contain nonempty train, val and test row-index lists')
    indices = [index for flag in FLAGS for index in result[flag]]
    if any(type(index) is not int for index in indices) or sorted(indices) != list(range(len(groups))):
        raise ValueError('Split row indices must cover every source row exactly once')
    group_sets = {flag: {groups[i] for i in result[flag]} for flag in FLAGS}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if group_sets[a] & group_sets[b]:
            raise ValueError(f'Formulation leakage between {a} and {b}')


def scaler_record(values):
    scaler = StandardScaler().fit(values)
    return {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}


class ElectrolyteDataset(Dataset):
    def __init__(self, indices, inputs, labels, masks, groups, tasks):
        self.indices = indices
        self.inputs, self.labels, self.masks = inputs, labels, masks
        self.groups, self.tasks = groups, tasks

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        return {'features': torch.tensor(self.inputs[row], dtype=torch.float32),
                'targets': {task: torch.tensor(self.labels[task][row], dtype=torch.float32)
                            for task in self.tasks},
                'target_masks': {task: torch.tensor(self.masks[task][row], dtype=torch.bool)
                                 for task in self.tasks},
                'sample_id': str(row), 'group_id': self.groups[row]}


def electrolyte_loaders(args, saved=None):
    root = Path(args.root_path).expanduser()
    path = root if root.is_file() else root / 'dataset' / 'experimental_data.json'
    if not path.is_file():
        path = root / 'experimental_data.json'
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if saved and digest != saved['source_sha256']:
        raise ValueError('Electrolyte source data changed since checkpoint creation')
    rows = json.loads(content)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Electrolyte data must be a nonempty JSON array of records')
    groups = [formulation_key(row) for row in rows]
    splits = saved['split_indices'] if saved else partition(groups, args)
    validate_partition(splits, groups)
    labels, masks = {}, {}
    for task in args.tasks:
        pairs = [target(row, task) for row in rows]
        labels[task] = np.asarray([value for value, _ in pairs])
        masks[task] = np.asarray([mask for _, mask in pairs])
        for flag in FLAGS:
            if not masks[task][splits[flag]].any():
                raise ValueError(f'No valid {task} labels in {flag}; check {task}_mask and task selection')
    eligible = {flag: [i for i in indices if any(masks[t][i] for t in args.tasks)]
                for flag, indices in splits.items()}
    inputs = np.stack([features(row, args.fingerprint_dim) for row in rows])
    feature_scaler = saved['feature_scaler'] if saved else scaler_record(inputs[eligible['train']])
    inputs = (inputs - np.asarray(feature_scaler['mean'])) / np.asarray(feature_scaler['scale'])
    if not np.isfinite(inputs).all():
        raise ValueError('Non-finite electrolyte input features')
    target_scalers = {}
    for task in args.tasks:
        train = eligible['train']
        observed = labels[task][train][masks[task][train]]
        if saved:
            record = saved['target_scalers'][task]
        else:
            fitted = scaler_record(observed.reshape(-1, 1))
            record = {'mean': fitted['mean'][0], 'scale': fitted['scale'][0]}
        target_scalers[task] = record
        labels[task] = np.where(masks[task], (labels[task] - record['mean']) / record['scale'], 0.0)
    state = {'protocol': 'formulation_group', 'source_sha256': digest,
             'source_path': str(path.resolve()), 'source_rows': len(rows),
             'representation': 'weighted SMILES character 1/2/3-gram SHA256 buckets, salt fraction, temperature',
             'group_definition': 'source SMILES + normalized component ratios + salt fraction; excludes temperature',
             'split_seed': args.split_seed, 'split_indices': splits,
             'feature_scaler': feature_scaler, 'target_scalers': target_scalers,
             'splits': {flag: {'files': [str(i) for i in eligible[flag]],
                               'samples': len(eligible[flag]),
                               'groups': len({groups[i] for i in eligible[flag]}),
                               'valid_labels': {task: int(masks[task][eligible[flag]].sum()) for task in args.tasks}}
                        for flag in FLAGS}}
    loaders = []
    for flag in FLAGS:
        dataset = ElectrolyteDataset(eligible[flag], inputs, labels, masks, groups, args.tasks)
        loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=flag == 'train',
                                  num_workers=args.num_workers, drop_last=False,
                                  generator=torch.Generator().manual_seed(args.seed)))
        print(f'{flag}: {len(dataset)} records, {state["splits"][flag]["groups"]} formulation groups')
    return tuple(loaders), state
