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
        weight[i:i+64], entorpy = higgs_quantize_dequantize(weight[i:i+64], edenn_d, edenn_n)
    weight = weight.reshape(weight.shape[0], mult, -1)
    
    # Cut the padded values
    weight = weight[...,:hadamard_groupsize]
    
    # Unscale
    weight = (weight * scales[:, :, None]).reshape(weight.shape[0], -1)

    class Bias(nn.Module):
        def __init__(self, bias):
            super().__init__()
            assert bias is not None
            self.bias = bias

        def forward(self, x):
            return x + self.bias

    output = HadLinear(weight.half(), hadamard_groupsize)

    if layer.bias is not None:
        output = nn.Sequential(output, Bias(layer.bias))

    return output, entorpy
    

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
        layer.to('meta')
        torch.cuda.empty_cache()
        
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
    if hasattr(model.model, 'rotary_emb'):
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


def get_zero_shots(model, task_list = ('arc_easy',), num_fewshots=1, batch_size=1):
    import lm_eval
    from transformers import AutoTokenizer

    lm_eval_model = lm_eval.models.huggingface.HFLM(
        pretrained=model,
        batch_size=batch_size,
        tokenizer=AutoTokenizer.from_pretrained(model.config._name_or_path, use_fast=False),
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


@torch.no_grad()
def eval_grid(edenn_d: int, edenn_n: int):
    x = torch.empty((2**16, edenn_d), device=DEV).normal_()
    dequant, entropy = higgs_quantize_dequantize(x, edenn_d, edenn_n)
    mse = (x - dequant).pow(2).mean().item()
    return mse, entropy / edenn_d


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
        '--slopes_tag', type=str, default="default",
        help='tag for wandb slopes values'
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
        '--method', type=str, choices=["rtn", "gptq"], default="gptq", help="Method to quantize with",
    )
    parser.add_argument(
        '--dataset', type=str, default='red', choices=['red'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=256,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--bits_sep', type=float, default=2.625,
        help='Bits separation value.'
    )
    parser.add_argument(
        '--blockwise', action='store_true',
        help='Enable blockwise quantization.'
    )
    parser.add_argument(
        '--target_bits', type=float, default=4.0,
        help='Target bits value.'
    )
    parser.add_argument(
        '--zeroshot-batch-size', type=int, required=False, default=1,
    )
    parser.add_argument(
        '--mmlu-batch-size', type=int, required=False, default=1,
    )
    parser.add_argument(
        '--skip_ppl_eval', action='store_true', help='Skip PPL evaluations.'
    )
    parser.add_argument(
        '--slopes_wandb_name', type=str, default='galqiwi/test',
        help='WandB name for slopes.'
    )

    args = parser.parse_args()

    wandb.init(
        # track hyperparameters and run metadata
        config=args,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cpu")
    model.seqlen = args.seqlen
    model.eval()

    config = get_config(args)
    real_bits = config['real_bits']
    predicted_ppl = config['predicted_ppl']
    optimal_config = config['optimal_config']
    wandb.log({
        'real_bits': real_bits,
        'predicted_ppl': predicted_ppl,
        'optimal_config': optimal_config,
    })


    match args.method:
        case "rtn":
            model = llama_rtn(model, optimal_config, args.hadamard_groupsize, DEV)
        case "gptq":
            dataloader, testloader = get_loaders(
                args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            model = llama_gptq(model, args.nsamples, dataloader, DEV, optimal_config, args.hadamard_groupsize)
        case _:
            raise Exception("AAA")

    model = model.half()

    if not args.skip_ppl_eval:
        datasets = ['wikitext2']
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            ppl = llama_eval(model, testloader, DEV)
            wandb.log({f'ppl_{dataset}': ppl})

    from parallel import dispatch_model_parallel
    model = dispatch_model_parallel(model)

    wandb.log(get_zero_shots(
        model,
        task_list=['winogrande','piqa','hellaswag', 'arc_easy','arc_challenge'],
        num_fewshots=1,
        batch_size=args.zeroshot_batch_size,
    ))
    wandb.log(
        filter_dict(
            get_zero_shots(model, task_list=['mmlu',], num_fewshots=5, batch_size=args.mmlu_batch_size),
            'mmlu@5'
        )
    )


if __name__ == '__main__':
    main()
