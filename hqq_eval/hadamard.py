from functools import cache, lru_cache
import math
import numpy as np
from scipy.special import erf

import torch
from torch import nn
import torch.nn.functional as F

from fast_hadamard_transform import hadamard_transform


# half-normal centroids
opt_hn_centroids = {
    1: [0.7978845608028654],
    2: [0.4527800398860679, 1.5104176087114887],
    3: [0.24509416307340598, 0.7560052489539643, 1.3439092613750225, 2.151945669890335],
    4: [
        0.12839501671105813,
        0.38804823445328507,
        0.6567589957631145,
        0.9423402689122875,
        1.2562309480263467,
        1.6180460517130526,
        2.069016730231837,
        2.732588804065177,
    ],
}


def section_variance(a, b, c) -> float:
    if math.isinf(c):
        return (
            np.sqrt(2 / np.pi) * np.exp(-(a**2) / 2) * (a - 2 * b)
            - (b**2 + 1) * erf(a / np.sqrt(2))
            + (b**2 + 1) * erf(c / np.sqrt(2))
        )
    else:
        return (
            np.sqrt(2 / np.pi) * np.exp(-(a**2) / 2) * (a - 2 * b)
            - (b**2 + 1) * erf(a / np.sqrt(2))
            + (b**2 + 1) * erf(c / np.sqrt(2))
            + np.sqrt(2 / np.pi) * np.exp(-(c**2) / 2) * (2 * b - c)
        )


def gen_boundaries(centroids):
    return [(a + b) / 2 for a, b in zip(centroids[:-1], centroids[1:])]


def gen_all_normal_quantization_constants():
    # add symmetric negative normal centroids
    centroids = {i: [-j for j in reversed(c)] + c for i, c in opt_hn_centroids.items()}

    # centroids to bin boundaries
    boundaries = {i: gen_boundaries(c) for i, c in centroids.items()}

    return centroids, boundaries
    

@cache
def bits_var():
    result = {0: 1}
    for bits, centers in opt_hn_centroids.items():
        borders = [0] + [(a + b) / 2 for a, b in zip(centers[:-1], centers[1:])] + [float("inf")]
        variance = sum(section_variance(a, b, c) for a, b, c in zip(borders[:-1], centers, borders[1:]))
        result[bits] = variance
    return result


@cache
def get_all_quantization_constants_tensors(device):
    centroids, boundaries = gen_all_normal_quantization_constants()

    centroids = {i: torch.tensor(c, device=device) for i, c in centroids.items()}
    boundaries = {i: torch.tensor(b, device=device) for i, b in boundaries.items()}

    return centroids, boundaries



centroids, boundaries = get_all_quantization_constants_tensors("cuda")

def quantize_hadamard(weight, bits):
    assignments = torch.bucketize(weight, boundaries[bits])
    return torch.take(centroids[bits], assignments)


def pad_to_block(tensor, dims, blocksize):
    pad_dims = [0 for _ in range(2 * len(tensor.shape))]
    for dim in dims:
        size = tensor.shape[dim]
        next_multiple_of_block = ((size - 1) // blocksize + 1) * blocksize
        delta = next_multiple_of_block - size
        pad_dims[-2 * dim - 1] = delta
    
    return F.pad(tensor, pad_dims, "constant", 0)


class HadLinear(nn.Module):
    def __init__(self, weight, blocksize, do_hadamard, actquant=16):
        super().__init__()
        self.blocksize = blocksize
        self.do_hadamard = do_hadamard
        self.actquant = actquant

        if do_hadamard:
            weight = weight / math.sqrt(blocksize)
        self.weight_dtype = weight.dtype
        out_dim, in_dim = weight.shape
        self.inner = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False, device='meta')
        self.inner.weight = nn.Parameter(
            weight,
            requires_grad=False,
        )

    def forward(self, input):
        if self.do_hadamard:
            input = pad_to_block(input, [-1], self.blocksize)
            mult = input.shape[-1] // self.blocksize
            input = input.reshape(input.shape[:-1] + (mult, self.blocksize))

            if self.actquant != 16:
                scale = torch.linalg.norm(input, axis=-1, keepdim=True)
                input = hadamard_transform(input) / scale
                input = quantize_hadamard(input, self.actquant)
                input = input * scale / math.sqrt(self.blocksize)
                input = input.to(self.weight_dtype)
            else:
                input = hadamard_transform(input, scale=1/math.sqrt(self.blocksize))

            input = input.reshape(input.shape[:-2] + (mult * self.blocksize,))
        else:
            if self.actquant != 16:
                raise NotImplementedError("AAA")
            else:
                pass

        return self.inner(input)
