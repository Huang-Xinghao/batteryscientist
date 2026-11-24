"""Measured SOH history -> future SOH; no life labels or charge-curve inputs."""
import hashlib
import math
import random
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .batterylife import cell_location
from .splits import cell_splits, training_sampler


class SOHSequenceDataset(Dataset):
    def __init__(self, args, names):
        self.cells, self.windows, self.records = [], [], []
        coordinate = args.cycle_coordinate
        for name in names:
            root = Path(args.root_path).expanduser().resolve()
            path = root / args.dataset / name
            if not path.is_file():
                path = root / name
            if not path.is_file():
                path = cell_location(root, name)[0]
            content = path.read_bytes()
            data = pickle.loads(content)
            nominal = float(data['nominal_capacity_in_Ah'])
            if not np.isfinite(nominal) or nominal <= 0:
                raise ValueError(f'Invalid nominal capacity in {name}')
            cycles = sorted(data['cycle_data'], key=lambda row: row['cycle_number'])
            raw = np.array([row['cycle_number'] for row in cycles], dtype=float)
            if not np.isfinite(raw).all() or len(np.unique(raw)) != len(raw):
                raise ValueError(f'Nonfinite or duplicate cycle numbers in {name}')
            values = []
            for row in cycles:
                capacity = np.asarray(row['discharge_capacity_in_Ah'], dtype=float)
                finite = capacity[np.isfinite(capacity)]
                values.append(float(finite.max())/nominal if len(finite) else np.nan)
            y = np.asarray(values, dtype=np.float32)
            cell_id = str(data.get('cell_id', name))
            index = len(self.cells)
            self.cells.append((name, cell_id, y, raw))
            count = 0
            for anchor in range(args.context, len(y)-args.horizon+1):
                start, end = anchor-args.context, anchor+args.horizon
                if not (np.isfinite(y[start:end]) & (y[start:end] > 0)).all():
                    continue
                if coordinate == 'cycle_number' and not np.allclose(np.diff(raw[start:end]), args.cycle_step,
                                                                    rtol=0, atol=1e-8):
                    continue
                lo, hi = getattr(args, 'eval_cycle_min', None), getattr(args, 'eval_cycle_max', None)
                position = raw[anchor-1] if coordinate == 'cycle_number' else anchor
                if (lo is not None and position < lo) or (hi is not None and position > hi):
                    continue
                self.windows.append((index, anchor)); count += 1
            self.records.append(dict(file=name, cell_id=cell_id, sha256=hashlib.sha256(content).hexdigest(),
                                     windows=count, nominal_capacity_in_Ah=nominal))
        if not self.windows:
            raise ValueError('No valid SOH history/forecast windows in split')
        if len({cell[1] for cell in self.cells}) != len(self.cells):
            raise ValueError('Duplicate cell_id in split')
        self.context, self.horizon = args.context, args.horizon

    def __len__(self):
        return len(self.windows)

    def window_cell_ids(self):
        return [self.cells[index][1] for index, _ in self.windows]

    def __getitem__(self, index):
        cell, anchor = self.windows[index]
        name, cell_id, y, raw = self.cells[cell]
        return dict(history=torch.from_numpy(y[anchor-self.context:anchor].copy()),
                    target=torch.from_numpy(y[anchor:anchor+self.horizon].copy()), cell_id=cell_id,
                    sample_id=f'{name}:{anchor}', current_record=anchor, raw_cycle_number=raw[anchor-1])


def sequence_loaders(args, state=None):
    splits, provenance = cell_splits(args)
    access = getattr(args, 'dataaccess', 100)
    if access < 100:
        splits = dict(splits)
        count = max(1, math.ceil(len(splits['train']) * access / 100))
        subset = set(random.Random(getattr(args, 'split_seed', 2021)).sample(splits['train'], count))
        splits['train'] = [name for name in splits['train'] if name in subset]
    datasets = [SOHSequenceDataset(args, splits[flag]) for flag in ('train', 'val', 'test')]
    seen = set()
    for dataset in datasets:
        ids = {cell[1] for cell in dataset.cells}
        if seen & ids:
            raise ValueError('Cell identity leakage between splits')
        seen.update(ids)
    record = dict(protocol='SOH history forecasting', split_source=provenance,
                  cycle_coordinate=args.cycle_coordinate, cycle_step=args.cycle_step,
                  context=args.context, horizon=args.horizon,
                  capacity_targets='max finite discharge capacity / nominal capacity',
                  preprocessing='SOH fractions; no fitted normalization',
                  splits={flag: dataset.records for flag, dataset in zip(('train', 'val', 'test'), datasets)})
    if state is not None and record != state:
        raise ValueError('SOH source data, splits or protocol changed since checkpoint')
    loaders = []
    for i, dataset in enumerate(datasets):
        sampler = training_sampler(dataset, args) if i == 0 else None
        loaders.append(DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                  shuffle=i == 0 and sampler is None, num_workers=args.num_workers,
                                  generator=torch.Generator().manual_seed(args.seed)))
    return tuple(loaders), record
