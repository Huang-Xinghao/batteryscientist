"""Domain plugins: data, task targets, model inputs and evaluation transforms."""

from .cell import CellAdapter
from .electrolyte import ElectrolyteAdapter

from .soh_sequence import SOHSequenceAdapter

ADAPTERS = {'cell': CellAdapter, 'electrolyte': ElectrolyteAdapter}
ADAPTER_REGISTRY = {'cell_curves': CellAdapter, 'soh_sequence': SOHSequenceAdapter,
                    'electrolyte': ElectrolyteAdapter}


def adapter_class(args):
    name = getattr(args, 'adapter', None)
    if name is None:
        return ADAPTERS[getattr(args, 'domain', 'cell')]
    cls = ADAPTER_REGISTRY[name]
    if cls.domain != args.domain:
        raise ValueError(f'Adapter {name!r} does not support domain {args.domain!r}')
    return cls


def make_adapter(args):
    domain = getattr(args, 'domain', 'cell')
    if domain not in ADAPTERS:
        raise ValueError(f'Unknown domain {domain!r}; choose from {tuple(ADAPTERS)}')
    return adapter_class(args)(args)
