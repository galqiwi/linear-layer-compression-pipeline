import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fast_hadamard_transform import hadamard_transform

import pathlib
grids_folder = pathlib.Path(__file__).parent.parent.resolve().joinpath("grids/")

GRIDS = {
}
# Read files in the folder and read grids in the EDEN{DIM}_{SIZE}.pt format
for file in grids_folder.iterdir():
    if file.suffix == ".pt":
        try:
            dim, size = map(int, file.stem[4:].split('-'))
        except ValueError:
            continue
        GRIDS[dim] = GRIDS.get(dim, {})
        GRIDS[dim][size] = torch.load(file)

GRID_NORMS = {k1: {k2: torch.linalg.norm(GRIDS[k1][k2], dim=1) ** 2 for k2 in v1.keys()} for k1, v1 in GRIDS.items()}


def entropy(idx):
    _, counts = torch.unique(idx, return_counts=True)
    counts = counts.to(torch.float)
    return -torch.sum(counts / len(idx) * torch.log2(counts / len(idx))).item()

def higgs_quantize(x, dim, size):
    assert size <= 256
    return torch.argmax(2 * x @ GRIDS[dim][size].T - GRID_NORMS[dim][size], dim=-1).to(torch.uint8)

def higgs_quantize_dequantize(x, dim, size):
    idx = torch.argmax(2 * x @ GRIDS[dim][size].T - GRID_NORMS[dim][size], dim=-1)
    return GRIDS[dim][size][idx], entropy(idx)


def pad_to_block(tensor, dims, had_block_size, value=0):
    pad_dims = [0 for _ in range(2 * len(tensor.shape))]
    for dim in dims:
        size = tensor.shape[dim]
        next_multiple_of_1024 = ((size - 1) // had_block_size + 1) * had_block_size
        delta = next_multiple_of_1024 - size
        pad_dims[-2 * dim - 1] = delta
    
    return F.pad(tensor, pad_dims, "constant", value)

class HadLinear(nn.Module):
    def __init__(self, weight, had_block_size=1024):
        super().__init__()
        self.register_buffer('had_block_size', torch.tensor(0))
        self.had_block_size = torch.tensor(had_block_size)
        weight = weight / math.sqrt(had_block_size)
        out_dim, in_dim = weight.shape
        self.inner = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False, device='meta')
        self.inner.weight = nn.Parameter(
            weight,
            requires_grad=False,
        )

    def forward(self, input):
        input = pad_to_block(input, [-1], self.had_block_size)
        mult = input.shape[-1] // self.had_block_size
        input = input.reshape(input.shape[:-1] + (mult, self.had_block_size))
        input = hadamard_transform(input, scale=1 / math.sqrt(self.had_block_size))
        input = input.reshape(input.shape[:-2] + (mult * self.had_block_size,))
        return self.inner(input)
