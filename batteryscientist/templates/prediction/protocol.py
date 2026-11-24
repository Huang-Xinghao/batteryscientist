"""Persist comparison conditions shared by candidate models, before scoring tests."""
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path


# Model, seed, optimizer and training schedule intentionally do not define the
# comparison population. The selected metric and input/target semantics do.
COMPARISON_KEYS = ('domain', 'dataset', 'tasks', 'context', 'horizon', 'cycle_coordinate',
                   'cycle_step', 'early_cycle_threshold', 'charge_discharge_length',
                   'eval_cycle_min', 'eval_cycle_max', 'aggregation', 'selection_metric')


def comparison_protocol(args, adapter):
    values = {key: getattr(args, key, None) for key in COMPARISON_KEYS}
    if type(adapter).__name__ == 'SOHSequenceAdapter':
        for key in ('early_cycle_threshold', 'charge_discharge_length'):
            values.pop(key, None)
    values['adapter'] = getattr(args, 'adapter', None) or type(adapter).__name__
    return {'schema_version': 1, 'comparison': values, 'data': adapter.state()}


def verify_protocol(args, adapter):
    """Atomically create once or compare; a different protocol needs another path."""
    path = getattr(args, 'protocol_file', None)
    if not path:
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = comparison_protocol(args, adapter)
    encoded = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
    with destination.with_suffix(destination.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            if json.loads(destination.read_text()) != record:
                raise ValueError(f'Comparison protocol changed: {destination}. '
                                 'Reuse its cell membership, windows and evaluation settings; '
                                 'use another protocol_file for an intentional different experiment.')
        else:
            descriptor, name = tempfile.mkstemp(prefix='.protocol-', dir=destination.parent)
            try:
                with os.fdopen(descriptor, 'w') as stream:
                    stream.write(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
                os.replace(name, destination)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
    return {'path': str(destination), 'sha256': hashlib.sha256(encoded.encode()).hexdigest()}
