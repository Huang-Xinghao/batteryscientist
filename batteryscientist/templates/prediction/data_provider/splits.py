"""Explicit cell membership shared by curve and history-sequence datasets."""
import hashlib
import json
from pathlib import Path


def cell_splits(args):
    from .batterylife import fixed_splits
    path = getattr(args, 'split_file', None)
    if not path:
        return fixed_splits(args.dataset, getattr(args, 'condition', 'all'))
    if getattr(args, 'condition', 'all') != 'all':
        raise ValueError('Use explicit filtered membership in split_file rather than condition')
    content = Path(path).read_bytes()
    source = json.loads(content)
    # Accept a single dataset or the multi-dataset format used by Data Report.
    result = source['datasets'][args.dataset] if 'datasets' in source else source
    if set(result) != {'train', 'val', 'test'}:
        raise ValueError('Cell split_file requires exactly train/val/test')
    seen = set()
    for flag, names in result.items():
        if not isinstance(names, list) or not names:
            raise ValueError(f'Empty or invalid cell split {flag}')
        for name in names:
            if not isinstance(name, str) or Path(name).name != name or not name.endswith('.pkl'):
                raise ValueError('Cell split members must be PKL filenames')
            if name in seen:
                raise ValueError(f'Duplicate cell or cell leakage: {name}')
            seen.add(name)
    return result, {'source': str(Path(path).resolve()), 'sha256': hashlib.sha256(content).hexdigest()}


def training_sampler(dataset, args):
    """Window shuffle by default; optional equal expected weight per cell."""
    if getattr(args, 'sampling', 'window') != 'cell_balanced':
        return None
    import torch
    from torch.utils.data import WeightedRandomSampler
    ids = dataset.window_cell_ids()
    from collections import Counter
    counts = Counter(ids)
    return WeightedRandomSampler([1.0/counts[cell] for cell in ids], len(ids), replacement=True,
                                 generator=torch.Generator().manual_seed(args.seed))
