"""Minimal history-sequence baseline; replace/register research models independently."""
from torch import nn


class Model(nn.Module):
    config_keys = ('context', 'horizon')

    def __init__(self, args):
        super().__init__()
        self.projection = nn.Linear(args.context, args.horizon)

    def forward(self, history):
        return self.projection(history)
