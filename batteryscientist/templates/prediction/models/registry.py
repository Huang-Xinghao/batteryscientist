"""Register model factories here; adapters declare compatible models."""

from .IC2ML import Model as IC2ML
from .CompositionMLP import Model as CompositionMLP
from .SOHLinear import Model as SOHLinear
from .SOHBaselines import Persistence, Trend, MLP, TCN

MODELS = {'IC2ML': IC2ML, 'CompositionMLP': CompositionMLP, 'SOHLinear': SOHLinear,
          'SOHPersistence': Persistence, 'SOHTrend': Trend, 'SOHMLP': MLP, 'SOHTCN': TCN}


def build_model(args):
    if args.model not in MODELS:
        raise ValueError(f'Unknown model {args.model!r}; choose from {tuple(MODELS)}')
    return MODELS[args.model](args).float()


def validate_model(args):
    """Model-specific constraints do not apply to every model in a domain."""
    if args.model not in MODELS:
        raise ValueError(f'Unknown model {args.model!r}')
    validator = getattr(MODELS[args.model], 'validate_config', None)
    if validator:
        validator(args)
