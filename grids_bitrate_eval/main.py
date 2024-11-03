import os
from typing import Optional

import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

import argparse
from datautils import *

import wandb
from tqdm import tqdm, trange

from get_config import get_config

from edenn import higgs_quantize_dequantize, pad_to_block, HadLinear
from fast_hadamard_transform import hadamard_transform
from gptq import apply_gptq, get_accumulate_input_fn
import torch
import requests
import os
import io


def get_af4_grid(block_size):
    url = (
            'https://github.com/galqiwi/linear-layer-compression-pipeline/raw/' +
            f'5dbca968897883435db856549a7d57da14ca14ae/2024-09-28/af4_{block_size}.pt'
    )
    return torch.load(io.BytesIO(requests.get(url).content))


def get_af3_grid(block_size):
    assert block_size == 64

    return torch.tensor([-1., -0.42985193, -0.13062135, 0., 0.09237868, 0.28947706, 0.53729404, 1.])


def get_nf3_grid(block_size):
    return torch.tensor([-1.0000, -0.4786, -0.2171, 0.0000, 0.1609, 0.3379, 0.5626, 1.0000])


import copy
import torch

import torch
import numpy as np


def get_int8_grid():
    return torch.tensor(list(np.linspace(-1, 1, 255)) + [float('+inf')])


NF4_CODES = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
    -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434,
    0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=torch.float16)


def get_closest_idx(x, grid):
    _grid_len, = grid.shape
    input_shape = x.shape
    x = x.reshape(-1)

    output = (x[:, None] - grid[None, :]).abs().min(dim=1).indices
    assert output.shape == x.shape

    return output.reshape(input_shape)


def quantize_weight(weight, block_size=64, codes=NF4_CODES):
    out_dim, in_dim = weight.shape

    codes = copy.deepcopy(codes).to(weight.device)

    weight_groups = weight.reshape(-1, block_size)

    scales = weight_groups.abs().max(dim=1).values

    assert scales.shape == (out_dim * in_dim // block_size,)
    weight_quantized = get_closest_idx(
        weight_groups / scales[:, None],
        codes,
    ).reshape(out_dim, in_dim).to(weight.device)

    return weight_quantized, scales


def dequantize_weight(weight_quantized, scales, block_size=64, codes=NF4_CODES):
    out_dim, in_dim = weight_quantized.shape

    codes = copy.deepcopy(codes).to(weight_quantized.device)

    return (
            codes[weight_quantized].reshape(-1, block_size) *
            scales[:, None]
    ).reshape(out_dim, in_dim)


def quantize_dequantize_weight(weight, block_size=64, codes=NF4_CODES):
    weight_quantized, scales = quantize_weight(weight, block_size=block_size, codes=codes)
    scales = scales.half()
    return dequantize_weight(weight_quantized, scales, block_size=block_size, codes=codes)


DEV = torch.device('cuda')


def filter_dict(x, inner):
    return {
        key: value
        for key, value in x.items()
        if inner.lower() in key.lower()
    }


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def replace_submodule(module, submodule_path, new_submodule):
    submodule_names = submodule_path.split(".")
    for submodule in submodule_names[:-1]:
        module = getattr(module, submodule)
    setattr(module, submodule_names[-1], new_submodule)


@torch.no_grad()
def quantize_linear_layer(layer: nn.Linear, hadamard_groupsize: int, edenn_d: int, edenn_n: int):
    weight = layer.weight.float()
    # Pad to Hadamard transform size
    weight = pad_to_block(weight, [1], hadamard_groupsize)

    # Scale and Hadamard transform
    mult = weight.shape[1] // hadamard_groupsize
    weight = weight.reshape(-1, mult, hadamard_groupsize)
    scales = torch.linalg.norm(weight, axis=-1)
    weight = hadamard_transform(weight) / scales[:, :, None]

    # Pad to edenn_d and project
    weight = pad_to_block(weight, [2], edenn_d).reshape(weight.shape[0], mult, -1, edenn_d)

    for i in range(0, weight.shape[0], 64):
        weight[i:i + 64], entorpy = higgs_quantize_dequantize(weight[i:i + 64], edenn_d, edenn_n)
    weight = weight.reshape(weight.shape[0], mult, -1)

    # Cut the padded values
    weight = weight[..., :hadamard_groupsize]

    # Unscale
    weight = (weight * scales[:, :, None]).reshape(weight.shape[0], -1)

    return HadLinear(weight.half(), hadamard_groupsize), entorpy


@torch.no_grad()
def llama_rtn(model, layerwise_edenn_config, hadamard_groupsize, device):
    linear_layers = find_layers(model)

    layer_names = sorted([
        layer_name
        for layer_name in linear_layers.keys()
        if 'lm_head' not in layer_name
    ])

    assert set(layer_names) == set(layerwise_edenn_config.keys())

    for layer_name in tqdm(layer_names, desc="Quantizing linear layers..."):
        layer = linear_layers[layer_name]
        edenn_d, edenn_n = layerwise_edenn_config[layer_name]

        if (edenn_d, edenn_n) == (-1, -1):
            continue

        quantized_layer, entropy = quantize_linear_layer(layer.to(device), hadamard_groupsize, edenn_d, edenn_n)
        replace_submodule(model, layer_name, quantized_layer.cpu())

    return model


@torch.no_grad()
def llama_gptq(model, nsamples, dataloader, dev, layerwise_edenn_config, hadamard_groupsize):
    assert False
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    outs = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            outs.append(torch.zeros_like(inp))
            attention_masks.append(kwargs['attention_mask'])
            position_ids.append(kwargs['position_ids'])
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    layer_counter = 0
    for i in trange(len(layers), desc="Quantizing with GPTQ..."):
        layer = layers[i].to(dev)
        linear_layers = find_layers(layer)

        hessians = {name: None for name in linear_layers}
        num_samples = {name: 0 for name in linear_layers}
        handles = [
            linear_layers[name].register_forward_hook(
                get_accumulate_input_fn(name, hessians, num_samples)
            ) for name in linear_layers
        ]
        for j in trange(nsamples, leave=False, desc="Before pass..."):
            outs[j] = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        for name, linear in linear_layers.items():
            (edenn_d, edenn_n) = layerwise_edenn_config[layer_counter]
            layer_counter += 1

            if (edenn_d, edenn_n) == (-1, -1):
                continue

            quantized_layer = apply_gptq(
                linear.weight.data, 2 * hessians[name] / num_samples[name],
                edenn_d=edenn_d, edenn_n=edenn_n,
                had_block_size=hadamard_groupsize,
            )

            quantized_linear = HadLinear(quantized_layer, hadamard_groupsize)
            replace_submodule(layer, name, quantized_linear)

        mse = 0
        norm = 0
        for j in trange(nsamples, leave=False, desc="After pass..."):
            out = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]
            mse += torch.nn.functional.mse_loss(outs[j][0], out[0]).item()
            norm += outs[j][0].float().pow(2).mean().item()
            inps[j] = out

        if any([inp.isnan().any() for inp in inps]):
            raise Exception("NaNs!")

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    assert layer_counter == 7 * 32

    model.config.use_cache = use_cache
    return model


@torch.no_grad()
def llama_eval(model, dataloader, dev):
    print('Evaluating ...')

    nsamples = len(dataloader)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_masks.append(kwargs['attention_mask'])
            position_ids.append(kwargs['position_ids'])
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    for i in trange(len(layers), desc=f"Evaluating layer-by-layer..."):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            inps[j] = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = (dataloader[i].to(dev))[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

    return ppl.item()


def get_module_by_path(model, path):
    if path == '':
        return model
    splitted = path.split('.', 1)
    if len(splitted) == 1:
        splitted.append('')
    next_name, suffix = splitted

    try:
        next_module = model[int(next_name)]
    except:
        next_module = getattr(model, next_name)

    return get_module_by_path(next_module, suffix)


def set_module_by_path(model, path, value):
    parts = path.split('.')
    prefix = '.'.join(parts[:-1])
    parent = get_module_by_path(model, prefix)
    setattr(parent, parts[-1], value)


def get_zero_shots(model, task_list=('arc_easy',), num_fewshots=1):
    import lm_eval

    lm_eval_model = lm_eval.models.huggingface.HFLM(
        pretrained=model,
    )

    tasks = lm_eval.tasks.get_task_dict(task_list)
    if num_fewshots != 1:
        # TODO: make fewshots properly
        for task_name in tasks:
            task = tasks[task_name]
            if isinstance(task, tuple):
                task = task[1]
            if task is None:
                continue
            task.config.num_fewshot = num_fewshots

    results = lm_eval.evaluator.evaluate(
        lm=lm_eval_model,
        task_dict=tasks,
    )

    result_dict = {task_name: task_result['acc,none'] for task_name, task_result in results['results'].items()}
    result_err_dict = {f'{task_name}_err': task_result['acc_stderr,none'] for task_name, task_result in
                       results['results'].items()}
    result_dict = dict(list(result_dict.items()) + list(result_err_dict.items()))

    if num_fewshots != 1:
        result_dict = {f'{task_name}@{num_fewshots}': acc for task_name, acc in result_dict.items()}

    return result_dict


from fast_hadamard_transform import hadamard_transform


class NoisyHadamarLinear(torch.nn.Module):
    def __init__(self, weight, bias, *, had_block_size=2048, relative_mse=0):
        super().__init__()

        weight = weight.detach().clone()
        if bias is not None:
            bias = bias.detach().clone()

        self.had_block_size = had_block_size

        self.out_features, self.in_features = weight.shape

        self.inner = torch.nn.Linear(self.in_features, self.out_features, bias=(bias is not None), dtype=weight.dtype,
                                     device=weight.device)

        assert self.in_features % self.had_block_size == 0, (self.in_features, self.had_block_size)
        weight = weight.reshape(self.out_features, self.in_features // self.had_block_size, self.had_block_size)
        weight = hadamard_transform(weight, scale=1 / (self.had_block_size ** 0.5))
        weight = weight.reshape(self.out_features, self.in_features)

        weight = weight + torch.randn_like(weight) * torch.norm(weight) * (relative_mse ** 0.5) / (
                    weight.numel() ** 0.5)

        self.inner.weight.data = weight
        if bias is not None:
            self.inner.bias.data = bias

    def forward(self, input):
        input_shape = input.shape

        assert input.shape[-1] % self.had_block_size == 0

        input = input.reshape(-1, self.had_block_size)
        input = hadamard_transform(input, scale=1 / (self.had_block_size ** 0.5))
        input = input.reshape(input_shape)

        return self.inner(input)


@torch.no_grad()
def eval_grid(edenn_d: int, edenn_n: int):
    x = torch.empty((2 ** 16, edenn_d), device=DEV).normal_()
    dequant, entropy = higgs_quantize_dequantize(x, edenn_d, edenn_n)
    mse = (x - dequant).pow(2).mean().item()
    return mse, entropy / edenn_d


def get_empty_config(layers):
    return {layer: (-1, -1) for layer in layers}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--tag', type=str, default="default",
        help='tag for wandb config'
    )
    parser.add_argument(
        '--hadamard_groupsize', type=int, default=1024, choices=[64, 128, 256, 512, 1024, 2048, 4096],
        help='Groupsize to use for hadamard; default is 1024.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--seqlen',
        type=int, default=8192, help='Seq len for PPL evals.'
    )
    parser.add_argument(
        '--dataset', type=str, default='red', choices=['red'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--grid', type=str, choices=["nf4", "af4", "af3", "nf3", "int8"], default="nf4", help="Grid to quantize with",
    )
    parser.add_argument(
        '--block_size',
        type=int, default=1024, help='Block size for quantization.'
    )
    parser.add_argument(
        '--do_hadamard',
        action='store_true', help='Do Hadamard transform.'
    )

    args = parser.parse_args()

    wandb.init(
        # track hyperparameters and run metadata
        config=args,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                 device_map="cpu")
    model.seqlen = args.seqlen
    model.eval()

    layers = sorted([
        layer for
        layer in find_layers(model).keys()
        if 'lm_head' not in layer
    ])

    model = model.half().cuda()

    if args.grid == "nf4":
        codes = NF4_CODES
    elif args.grid == "af4":
        codes = get_af4_grid(args.block_size)
    elif args.grid == "af3":
        codes = get_af3_grid(args.block_size)
    elif args.grid == "nf3":
        codes = get_nf3_grid(args.block_size)
    elif args.grid == "int8":
        codes = get_int8_grid()
    else:
        raise ValueError(f"Unknown grid {args.grid}")

    for layer in layers:
        linear = get_module_by_path(model, layer)

        if args.do_hadamard:
            new_linear = NoisyHadamarLinear(linear.weight, linear.bias, had_block_size=args.hadamard_groupsize)
            new_linear.inner.weight.data = quantize_dequantize_weight(new_linear.inner.weight, codes=codes.half(),
                                                                      block_size=args.block_size).cuda()
            set_module_by_path(model, layer, new_linear)
            continue

        linear.weight.data = quantize_dequantize_weight(linear.weight, codes=codes.half(),
                                                        block_size=args.block_size).cuda()

    model = model.half()

    datasets = ['wikitext2']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        ppl = llama_eval(model, testloader, DEV)
        wandb.log({f'ppl_wikitext2': ppl})

    model = model.to(DEV)
    wandb.log(get_zero_shots(model, task_list=['winogrande', 'piqa', 'hellaswag', 'arc_easy', 'arc_challenge'],
                             num_fewshots=1))
    wandb.log(
        filter_dict(
            get_zero_shots(model, task_list=['mmlu', ], num_fewshots=5),
            'mmlu@5'
        )
    )


if __name__ == '__main__':
    main()
