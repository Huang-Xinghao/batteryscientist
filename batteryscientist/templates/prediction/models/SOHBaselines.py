"""Small, explicit SOH forecasting baselines sharing exactly the same input windows.

These are local reference implementations, not reproductions of named papers.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Persistence(nn.Module):
    config_keys = ('context', 'horizon')

    def __init__(self, args):
        super().__init__()
        self.horizon = args.horizon

    def forward(self, history):
        return history[:, -1:].expand(-1, self.horizon)


class Trend(nn.Module):
    """OLS line over the observed history, extrapolated at future positions."""
    config_keys = ('context', 'horizon')

    def __init__(self, args):
        super().__init__()
        if args.context < 2:
            raise ValueError('SOHTrend needs context >= 2')
        x = torch.arange(args.context, dtype=torch.float32)
        self.register_buffer('centered', x-x.mean())
        self.register_buffer('future', torch.arange(args.context, args.context+args.horizon)-x.mean())

    def forward(self, history):
        slope = (history*self.centered).sum(1, keepdim=True)/self.centered.square().sum()
        return history.mean(1, keepdim=True) + slope*self.future


class MLP(nn.Module):
    config_keys = ('context', 'horizon', 'hidden_dim')

    def __init__(self, args):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(args.context, args.hidden_dim), nn.GELU(),
                                     nn.Linear(args.hidden_dim, args.horizon))

    def forward(self, history):
        return self.network(history)


class CausalBlock(nn.Module):
    def __init__(self, incoming, hidden, dilation):
        super().__init__()
        self.padding = 2*dilation
        self.conv1 = nn.Conv1d(incoming, hidden, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, dilation=dilation)
        self.skip = nn.Conv1d(incoming, hidden, 1) if incoming != hidden else nn.Identity()

    def forward(self, x):
        y = F.gelu(self.conv1(F.pad(x, (self.padding, 0))))
        y = F.gelu(self.conv2(F.pad(y, (self.padding, 0))))
        return y+self.skip(x)


class TCN(nn.Module):
    """Three causal residual blocks (dilations 1/2/4), direct multi-step head."""
    config_keys = ('context', 'horizon', 'hidden_dim')

    def __init__(self, args):
        super().__init__()
        h = args.hidden_dim
        self.encoder = nn.Sequential(CausalBlock(1, h, 1), CausalBlock(h, h, 2), CausalBlock(h, h, 4))
        self.head = nn.Linear(h, args.horizon)

    def forward(self, history):
        return self.head(self.encoder(history[:, None, :])[:, :, -1])
