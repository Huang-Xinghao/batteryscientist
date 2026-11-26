"""BatteryLife PKLs, published life labels and fixed cell-level splits for tempalte."""

from bisect import bisect_right
import json
import hashlib
import math
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

from .batterylife_preprocessing import BatteryLifePreprocessor

SPLIT_PATH = Path(__file__).with_name('batterylife_splits.json')
TASKS = ('soh', 'trajectory', 'rul', 'life')
NEED_KEYS = ('current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah',
             'discharge_capacity_in_Ah', 'time_in_s')


def selected_tasks(args):
    tasks = tuple(dict.fromkeys(getattr(args, 'tasks', ('soh', 'trajectory', 'rul'))))
    if not tasks or set(tasks) - set(TASKS):
        raise ValueError(f'tasks must be a nonempty selection of {TASKS}')
    return tasks


def fixed_splits(dataset, condition='all', split_path=SPLIT_PATH):
    """Keep BatteryLife membership and ordering, including chemistry-only subsets."""
    source = json.loads(Path(split_path).read_text())
    chemistry = {'NCA': 'Tongji1_', 'NCM': 'Tongji2_', 'NCMNCA': 'Tongji3_'}
    name = 'Tongji' if dataset in chemistry else dataset
    if name not in source['splits']:
        raise ValueError(f'Unknown dataset {dataset!r}; available: {sorted(source["splits"])}')
    result = {}
    for flag, names in source['splits'][name].items():
        if dataset in chemistry:
            names = [n for n in names if n.startswith(chemistry[dataset])]
        if condition != 'all':
            if name != 'Tongji':
                raise ValueError('condition filtering is supported only for Tongji')
            names = [n for n in names if n.split('_', 1)[1].rsplit('--', 1)[0] == condition]
        result[flag] = list(names)
    for flag in ('train', 'val', 'test'):
        if len(result[flag]) != len(set(result[flag])):
            raise ValueError(f'Duplicate cell in {flag} split')
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if set(result[a]) & set(result[b]):
            raise ValueError(f'Cell leakage between {a} and {b}')
    return result, {k: v for k, v in source.items() if k != 'splits'}


def cell_location(root, file_name):
    """Match BatteryLife's filename-prefix directory and label-file conventions."""
    prefix = file_name.split('_')[0]
    if prefix.startswith('Tongji'):
        directory, labels, label_key = 'Tongji', 'Tongji', file_name.replace('--', '-#')
    elif prefix == 'SMICH':
        directory, labels, label_key = 'MICH_EXP', 'SMICH', file_name
    else:
        directory = {'UL-PUR': 'UL_PUR', 'ISU-ILCC': 'ISU_ILCC',
                     'MICH': 'total_MICH'}.get(prefix, prefix)
        labels = 'total_MICH' if prefix == 'MICH' else prefix
        label_key = file_name
    actual_name = file_name[1:] if prefix == 'SMICH' else file_name
    path = root / directory / actual_name
    # BatteryLife copies MICH first, then MICH_EXP into total_MICH. Resolve
    # the same precedence without creating directories in the source dataset.
    if directory == 'total_MICH' and not path.is_file():
        expanded = root / 'MICH_EXP' / actual_name
        path = expanded if expanded.is_file() else root / 'MICH' / actual_name
    return path, root / 'Life labels' / f'{labels}_labels.json', label_key


class BatteryLifeDataset(Dataset):
    def __init__(self, args, flag, files, label_scaler=None):
        self.args, self.flag, self.files = args, flag, list(files)
        self.tasks = selected_tasks(args)
        self.context, self.horizon = args.context, args.horizon
        self.early_cycle_threshold = args.early_cycle_threshold
        self.cells, self.ends, self.cell_records = [], [], []
        preprocessor = BatteryLifePreprocessor(args.charge_discharge_length)
        root = Path(args.root_path).expanduser().resolve()
        label_cache = {}
        for file_name in self.files:
            path, labels_path, label_key = cell_location(root, file_name)
            if labels_path not in label_cache:
                label_cache[labels_path] = json.loads(labels_path.read_text())
            record = {'file': file_name, 'label_file': str(labels_path), 'label_key': label_key}
            self.cell_records.append(record)
            eol = label_cache[labels_path].get(label_key)
            # Same label eligibility as BatteryLife; never fabricate censored labels.
            if eol is None:
                record.update(status='excluded', reason='missing_published_life_label')
                continue
            if not isinstance(eol, (int, float)) or not np.isfinite(eol) or eol <= 0:
                raise ValueError(f'Invalid published life label for {file_name}: {eol}')
            record['life_label'] = float(eol)
            if eol <= self.early_cycle_threshold:
                record.update(status='excluded', reason='life_not_above_early_cycle_threshold')
                continue
            content = path.read_bytes()
            record['source_sha256'] = hashlib.sha256(content).hexdigest()
            data = pickle.loads(content)
            nominal = (1.85 if file_name.startswith('RWTH') else
                       3.2 if file_name.startswith('SNL_18650_NCA_25C_20-80') else
                       float(data['nominal_capacity_in_Ah']))
            if not np.isfinite(nominal) or nominal <= 0:
                raise ValueError(f'Invalid nominal capacity for {file_name}')
            raw_cycles = data['cycle_data']
            history_count = min(len(raw_cycles), self.early_cycle_threshold)
            target_extra = self.horizon if 'trajectory' in self.tasks else 0
            needed = min(len(raw_cycles), history_count + target_extra)
            frames, soh, original_cycles = [], [], []
            for index, cycle in enumerate(raw_cycles[:needed]):
                # BatteryLife numbers saved records 1..N, and fills invalid
                # capacities within each cycle only (never from a future cycle).
                frame = pd.DataFrame({key: cycle[key] for key in NEED_KEYS})
                frame['cycle_number'] = index + 1
                frame.loc[frame['charge_capacity_in_Ah'] < 0] = np.nan
                frame.loc[frame['discharge_capacity_in_Ah'] < 0] = np.nan
                frame.bfill(inplace=True)
                frames.append(frame)
                soh.append(frame['discharge_capacity_in_Ah'].max() / nominal)
                original_cycles.append(cycle.get('cycle_number', index + 1))
            if history_count < self.context:
                record.update(status='excluded', reason='insufficient_history', windows=0)
                continue
            frame = pd.concat(frames, ignore_index=True)
            try:
                curves = preprocessor.get_charge_discharge_curves(
                    file_name, frame, history_count, nominal)
            except (IndexError, ValueError) as exc:
                raise ValueError(f'BatteryLife curve preprocessing failed for {file_name}: {exc}') from exc
            curves = curves.astype(np.float32)
            soh = np.asarray(soh, dtype=np.float32)
            if not np.isfinite(curves).all():
                raise ValueError(f'Non-finite BatteryLife input curves in {file_name}')
            anchors = []
            for anchor in range(self.context, history_count + 1):
                if anchor >= eol:
                    break
                if flag != 'train':
                    lower, upper = getattr(args, 'eval_cycle_min', None), getattr(args, 'eval_cycle_max', None)
                    if lower is not None and anchor < lower or upper is not None and anchor > upper:
                        continue
                if 'trajectory' in self.tasks and (anchor + self.horizon > needed
                                                  or anchor + self.horizon >= eol):
                    continue
                if 'soh' in self.tasks and not np.isfinite(soh[anchor-self.context:anchor]).all():
                    raise ValueError(f'Invalid SOH targets in {file_name} at record {anchor}')
                if 'trajectory' in self.tasks and not np.isfinite(soh[anchor:anchor+self.horizon]).all():
                    raise ValueError(f'Invalid trajectory targets in {file_name} at record {anchor}')
                anchors.append(anchor)
            if not anchors:
                record.update(status='excluded', reason='no_eligible_task_windows', windows=0)
                continue
            self.cells.append({'file': file_name, 'curves': curves, 'soh': soh,
                               'life': float(eol), 'anchors': anchors,
                               'original_cycle_numbers': original_cycles})
            self.ends.append((self.ends[-1] if self.ends else 0) + len(anchors))
            record.update(status='included', windows=len(anchors), nominal_capacity_in_Ah=nominal,
                          observed_records=len(raw_cycles), first_anchor=anchors[0], last_anchor=anchors[-1])
        if not self.cells:
            reasons = {record['file']: record.get('reason') for record in self.cell_records}
            raise ValueError(f'No eligible {flag} samples for tasks {self.tasks}: {reasons}')
        self.unique_labels = np.asarray([cell['life'] for cell in self.cells])
        if flag == 'train' and label_scaler is None:
            label_scaler = StandardScaler().fit(self.unique_labels.reshape(-1, 1))
        if label_scaler is None:
            raise ValueError('Validation/test must reuse the training life-label scaler')
        self.label_scaler = label_scaler
        for cell in self.cells:
            cell['scaled_life'] = float(label_scaler.transform([[cell['life']]])[0, 0])

    def window_cell_ids(self):
        return [cell['file'] for cell in self.cells for _ in cell['anchors']]

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        cell_index = bisect_right(self.ends, index)
        start = self.ends[cell_index-1] if cell_index else 0
        cell = self.cells[cell_index]
        anchor = cell['anchors'][index-start]
        return {
            'cycle_curves': torch.from_numpy(cell['curves'][anchor-self.context:anchor].copy()),
            'history_soh': torch.from_numpy(cell['soh'][anchor-self.context:anchor].copy()),
            'future_soh': torch.from_numpy(cell['soh'][anchor:anchor+self.horizon].copy())
                          if 'trajectory' in self.tasks else torch.empty(0),
            # Direct BatteryLife total-life supervision. RUL = life - current_cycle
            # uses the benchmark's record coordinate, not the raw PKL cycle_number.
            'life_label': torch.tensor(cell['scaled_life'], dtype=torch.float32),
            'life': torch.tensor(cell['life'], dtype=torch.float32),
            'current_cycle': torch.tensor(anchor, dtype=torch.int64),
            'raw_cycle_number': torch.tensor(float(cell['original_cycle_numbers'][anchor-1])),
            'cell_id': cell['file'],
        }

    def manifest(self):
        return {'files': self.files, 'cells': self.cell_records, 'samples': len(self),
                'included_cells': len(self.cells)}


def batterylife_trainloader(args, label_scaler=None):
    tasks = selected_tasks(args)
    if args.context < 2 or args.early_cycle_threshold < args.context:
        raise ValueError('tempalte requires 2 <= context <= early_cycle_threshold')
    if 'trajectory' in tasks and args.horizon < 1:
        raise ValueError('trajectory requires a positive horizon')
    if args.charge_discharge_length < 4 or args.charge_discharge_length % 2:
        raise ValueError('charge_discharge_length must be even and >= 4')
    from .splits import cell_splits
    splits, provenance = cell_splits(args)
    empty = [flag for flag, files in splits.items() if not files]
    if empty:
        raise ValueError(f'BatteryLife fixed split for {args.dataset!r} with condition '
                         f'{getattr(args, "condition", "all")!r} has no cells in {empty}. '
                         'Use the full dataset (e.g. Tongji) or a subset with all three splits; '
                         'cells are never reassigned across benchmark splits.')
    access = getattr(args, 'dataaccess', 100)
    if not 0 < access <= 100:
        raise ValueError('dataaccess must be in (0, 100]')
    if access < 100:
        count = max(1, math.ceil(len(splits['train']) * access / 100))
        subset = set(random.Random(getattr(args, 'split_seed', 2021)).sample(splits['train'], count))
        splits['train'] = [name for name in splits['train'] if name in subset]
    train = BatteryLifeDataset(args, 'train', splits['train'], label_scaler)
    val = BatteryLifeDataset(args, 'val', splits['val'], train.label_scaler)
    test = BatteryLifeDataset(args, 'test', splits['test'], train.label_scaler)
    loaders = []
    for flag, dataset in zip(('train', 'val', 'test'), (train, val, test)):
        from .splits import training_sampler
        sampler = training_sampler(dataset, args) if flag == 'train' else None
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            shuffle=flag == 'train' and sampler is None,
                            num_workers=getattr(args, 'num_workers', 0),
                            drop_last=flag == 'train' and len(dataset) >= args.batch_size,
                            generator=torch.Generator().manual_seed(args.seed))
        loaders.append(loader)
        print(f'{flag}: {len(dataset.cells)}/{len(dataset.files)} cells, {len(dataset)} windows')
    train.split_provenance = provenance
    return tuple(loaders)
