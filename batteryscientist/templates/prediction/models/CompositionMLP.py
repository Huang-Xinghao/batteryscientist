"""A composition-feature regression baseline; not the Bamboo-mixer architecture."""

from torch import nn


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 * args.fingerprint_dim + 2, args.hidden_dim),
            nn.GELU(), nn.LayerNorm(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim), nn.GELU(),
        )
        self.heads = nn.ModuleDict({task: nn.Linear(args.hidden_dim, 1) for task in args.tasks})

    def forward(self, features):
        embedding = self.encoder(features)
        return {task: head(embedding).squeeze(-1) for task, head in self.heads.items()}
