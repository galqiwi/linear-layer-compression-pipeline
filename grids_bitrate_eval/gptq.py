import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from edenn import higgs_quantize_dequantize, pad_to_block
from fast_hadamard_transform import hadamard_transform

from tqdm.auto import tqdm, trange


@torch.no_grad()
def gptq_block(block_weight: Tensor, block_hessian_inverse: Tensor, edenn_d: int, edenn_n: int) -> tuple[Tensor, Tensor]:
    quantized_block_weight = torch.zeros_like(block_weight)
    scaled_block_error = torch.zeros_like(block_weight)

    # Interate over the block's columns
    assert block_weight.shape[1] % edenn_d == 0
    for i in range(0, block_weight.shape[1], edenn_d):
        # Get the column and the corresponding inverse Hessian
        column_weight = block_weight[:, i:i+edenn_d]
        column_hessian_inverse = block_hessian_inverse[i, i]

        # Quantize the column weight
        quantized_column_weight, _ = higgs_quantize_dequantize(column_weight, edenn_d, edenn_n)
        quantized_block_weight[:, i:i+edenn_d] = quantized_column_weight.clone()
        dequantized_column_weight = quantized_column_weight

        # Update all the following columns within the block
        scaled_column_error = (column_weight - dequantized_column_weight) / column_hessian_inverse
        block_weight[:, i+1:] -= scaled_column_error.matmul(block_hessian_inverse[i:i+edenn_d, i+1:])
        scaled_block_error[:, i:i+edenn_d] = scaled_column_error

    return quantized_block_weight, scaled_block_error, block_weight


def prepare_inverse_hessian(hessian: Tensor, percdamp: float) -> Tensor:
    """Precomputes inverse Hessian
    Args:
        hessian (Tensor): problem hessian
        percdamp (float): diagonal damping constant for numerical stability
    Returns:
        Tensor: precomputed inverse Hessian
    """
    damp = percdamp * torch.mean(torch.diag(hessian))
    diag = torch.arange(hessian.shape[0], device=hessian.device)
    hessian[diag, diag] += damp
    hessian = torch.linalg.cholesky(hessian)
    hessian = torch.cholesky_inverse(hessian)
    hessian = torch.linalg.cholesky(hessian, upper=True)
    return hessian


def pad_to_block(tensor, dims, had_block_size, value=0):
    pad_dims = [0 for _ in range(2 * len(tensor.shape))]
    for dim in dims:
        size = tensor.shape[dim]
        next_multiple_of_1024 = ((size - 1) // had_block_size + 1) * had_block_size
        delta = next_multiple_of_1024 - size
        pad_dims[-2 * dim - 1] = delta
    
    return F.pad(tensor, pad_dims, "constant", value)


@torch.no_grad()
def apply_gptq(
    weight: torch.Tensor, hessian: torch.Tensor, edenn_d:int, edenn_n:int,
    had_block_size:int=1024, percdamp:float=.01
) -> tuple[Tensor, Tensor, Tensor]:
    blocksize = edenn_d
    while blocksize < 128:
        blocksize *= 2

    dtype = weight.dtype
    weight = weight.float()
    num_columns = weight.shape[1]
    hessian = hessian.float()

    # Normalize

    # scales = torch.linalg.norm(weight, axis=-1)
    weight = pad_to_block(weight, [1], had_block_size, value=0.01)
    hessian = pad_to_block(hessian, [0, 1], had_block_size)
    
    mult = weight.shape[1] // had_block_size
    weight = weight.reshape(-1, mult, had_block_size)
    hessian = hessian.reshape(mult, had_block_size, mult, had_block_size)
            
    weight = hadamard_transform(weight)
    
    scales = torch.linalg.norm(weight, axis=-1) / math.sqrt(had_block_size)
    weight = weight / scales[:, :, None]
    
    hessian = hadamard_transform(
        hadamard_transform(hessian, scale=1/math.sqrt(had_block_size)).permute(2, 3, 0, 1),
        scale=1/math.sqrt(had_block_size)
    ).permute(2, 3, 0, 1)
    
    # Pad to edenn_d
    weight = pad_to_block(weight, [2], edenn_d).reshape(weight.shape[0], -1)
    hessian = pad_to_block(hessian, [1, 3], edenn_d, 0).reshape(weight.shape[1], weight.shape[1])

    # Process the Hessian to obtain the precomputed inverse Hessian
    hessian_inverse = prepare_inverse_hessian(hessian, percdamp)

    # Iterate over the columns in blockss
    for block_start in trange(0, num_columns, blocksize, leave=False, desc="GPTQ blocks..."):
        # YOUR CODE HERE>>>>>>>>>
        block_end = min(block_start + blocksize, weight.shape[1])

        # Get the next block and quantize it
        quantized_block_weight, block_error, weight[:, block_start:block_end] = gptq_block(
            weight[:, block_start:block_end],
            hessian_inverse[block_start:block_end, block_start:block_end],
            edenn_d=edenn_d, edenn_n=edenn_n,
        )

        # Tune all the following blocks to mitigate the quantization error
        weight[:, block_start:block_end] = quantized_block_weight.clone()
        weight[:, block_end:] -= block_error.matmul(hessian_inverse[block_start:block_end, block_end:])
        # <<<<<<<<<<<<<<<<<<<<<<<
        
    # Cut padded weights
    weight = weight.reshape(weight.shape[0], mult, -1)[...,:had_block_size]
        
    weight = (weight * scales[:, :, None]).reshape(weight.shape[0], -1)
    return weight.to(dtype)


def get_accumulate_input_fn(name: str, hessians: Mapping[str, Tensor], num_samples: Mapping[str, int]):
    """Generate a callback that updates the corresponding hessians and counts when given input
    Args:
        name (str): module name
        hessians (Mapping[str, Tensor]): a dict of modules' hessians, accessible by module name
        num_samples (Mapping[str, int]): a dict of callback call counters
    """
    def tmp(_, inp, out):
        inp = inp[0].data # ... x hidden_size
        inp = inp.reshape((-1, inp.shape[-1])) # inputs x hidden_size
        inp = inp.t().float() # hidden_size x inputs
        num_samples[name] += inp.shape[1]
        if hessians[name] is None:
            hessians[name] = inp.matmul(inp.t())
        else:
            hessians[name] += inp.matmul(inp.t())
    return tmp