"""Small task-local decision alignment, not backbone retraining or future lifting."""
import torch
from torch import nn


class Aligner(nn.Module):
    def __init__(self, dim, kind, bound=0.2):
        super().__init__()
        self.kind, self.bound = kind, bound
        if kind == 'fusion':
            self.body = nn.Identity()
            width = dim
        else:
            width = 64
            self.project = nn.Sequential(nn.Linear(dim, width), nn.LayerNorm(width), nn.GELU())
            if kind == 'relation':
                layer = nn.TransformerEncoderLayer(width, 4, 128, dropout=0,
                                                   activation='gelu', batch_first=True)
                self.body = nn.TransformerEncoder(layer, 2)
            elif kind == 'mlp':
                self.body = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                          nn.Linear(width, width), nn.GELU())
            else:
                raise ValueError(kind)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, native):
        h = self.body(x if self.kind == 'fusion' else self.project(x))
        return native + self.bound * self.head(h).squeeze(-1).tanh()
