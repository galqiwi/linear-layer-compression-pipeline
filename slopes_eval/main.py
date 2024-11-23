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

from edenn import higgs_quantize_dequantize, pad_to_block, HadLinear
from fast_hadamard_transform import hadamard_transform
from gptq import apply_gptq, get_accumulate_input_fn

DEV = torch.device('cuda')


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_submodule(module, submodule_path):
    submodule_names = submodule_path.split(".")
    for submodule in submodule_names:
        module = getattr(module, submodule)
    return module


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
        replace_submodule(model, layer_name, quantized_layer.cpu().half())

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


def shallow_module_copy(model):
    import copy
    shared_embeddings = model.model.embed_tokens.weight is model.lm_head.weight
    model_state_dict = dict(model.named_parameters())
    model_state_dict_buffers = dict(model.named_buffers())
    model.to('meta')
    new_model = copy.deepcopy(model)
    if shared_embeddings:
        assert 'lm_head.weight' not in model_state_dict.keys()
        model_state_dict['lm_head.weight'] = model_state_dict['model.embed_tokens.weight']
    for param_name, param in model_state_dict.items():
        replace_submodule(new_model, param_name, nn.Parameter(data=param.data, requires_grad=False))
        replace_submodule(model, param_name, nn.Parameter(data=param.data, requires_grad=False))
    for name, buffer in model_state_dict_buffers.items():
        parent_name = '.'.join(name.split('.')[:-1])
        buffer_name = name.split('.')[-1]
        get_submodule(model, parent_name).register_buffer(buffer_name, buffer)
        get_submodule(new_model, parent_name).register_buffer(buffer_name, buffer)

    for param in model.parameters():
        assert param.device != torch.device('meta')
    for param in model.buffers():
        assert param.device != torch.device('meta')

    return new_model


@torch.no_grad()
def get_model_hidden_states(model, dataloader, dev):
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
        layers[i] = layer.to('meta')
        del layer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    hidden_states = inps

    model.config.use_cache = use_cache

    return hidden_states


@torch.no_grad()
def get_ppl(model, dataloader, hidden_states):
    dev = hidden_states[0].device
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    nsamples = len(dataloader)

    nlls = []
    for i in range(nsamples):
        sample_hidden_states = hidden_states[i]
        if model.model.norm is not None:
            sample_hidden_states = model.model.norm(sample_hidden_states)
        lm_logits = model.lm_head(sample_hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = (dataloader[i].to(dev))[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return ppl.item()


@torch.no_grad()
def llama_eval(model, dataloader, dev):
    print('Evaluating ...')

    hidden_states = get_model_hidden_states(model, dataloader, dev)
    output = get_ppl(model, dataloader, hidden_states)

    model.to('meta')

    return output


@torch.no_grad()
def get_kl_div(orig_model, model, dataloader, orig_hidden_states, hidden_states):
    nsamples = len(dataloader)
    dev = hidden_states[0].device

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)
    if orig_model.model.norm is not None:
        orig_model.model.norm = orig_model.model.norm.to(dev)
    orig_model.lm_head = orig_model.lm_head.to(dev)

    loss = 0
    for i in range(nsamples):
        sample_hidden_states = hidden_states[i]
        orig_sample_hidden_states = orig_hidden_states[i]
        if model.model.norm is not None:
            sample_hidden_states = model.model.norm(sample_hidden_states)
        if orig_model.model.norm is not None:
            orig_sample_hidden_states = orig_model.model.norm(orig_sample_hidden_states)
        lm_logits = model.lm_head(sample_hidden_states)
        orig_lm_logits = orig_model.lm_head(orig_sample_hidden_states)

        loss += kl_div_from_logits(inp=lm_logits, target=orig_lm_logits).item()

    return loss / nsamples


ORIG_MODEL_HIDDEN_STATES = None


@torch.no_grad()
def llama_eval_cross_entropy(orig_model, model, dataloader, dev):
    global ORIG_MODEL_HIDDEN_STATES
    print('Evaluating ...')

    if ORIG_MODEL_HIDDEN_STATES is None:
        ORIG_MODEL_HIDDEN_STATES = get_model_hidden_states(orig_model, dataloader, dev)
    orig_hidden_states = [x.detach().clone() for x in ORIG_MODEL_HIDDEN_STATES]
    hidden_states = get_model_hidden_states(model, dataloader, dev)

    return get_kl_div(
        orig_model=orig_model,
        model=model,
        dataloader=dataloader,
        orig_hidden_states=orig_hidden_states,
        hidden_states=hidden_states,
    )


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


@torch.no_grad()
def kl_div_from_logits(inp, target):
    assert inp.shape == target.shape
    n_tokens = inp.shape[-1]
    inp = inp.reshape(-1, n_tokens)
    target = target.reshape(-1, n_tokens)

    inp = torch.nn.functional.log_softmax(inp, dim=-1)
    target = torch.nn.functional.log_softmax(target, dim=-1)

    return torch.nn.functional.kl_div(input=inp, target=target, reduction='sum', log_target=True)


@torch.no_grad()
def eval_grid(edenn_d: int, edenn_n: int):
    x = torch.empty((2 ** 16, edenn_d), device=DEV).normal_()
    dequant, entropy = higgs_quantize_dequantize(x, edenn_d, edenn_n)
    mse = (x - dequant).pow(2).mean().item()
    return mse, entropy / edenn_d


def eval_ppl_by_config(args, model, layerwise_edenn_config):
    orig_model = None
    if args.div_loss:
        #     orig_model = copy.deepcopy(model)
        orig_model = shallow_module_copy(model)

    # model = copy.deepcopy(model)

    model = shallow_module_copy(model)
    model = llama_rtn(model, layerwise_edenn_config, args.hadamard_groupsize, DEV)

    # model = model.half()
    # if args.div_loss:
    #     orig_model = orig_model.half()

    datasets = ['random'] if args.div_loss else ['wikitext2_trimmed']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        if args.div_loss:
            ppl = llama_eval_cross_entropy(orig_model, model, testloader, DEV)
        else:
            ppl = llama_eval(model, dataloader, DEV)

        model.to('meta')
        if orig_model is not None:
            model.to('meta')

        return ppl


def get_empty_config(layers):
    return {layer: (-1, -1) for layer in layers}


###

def get_git_root(path):
    import os
    if os.path.dirname(path) == path:
        assert False, '.git not found'
    if os.path.isdir(os.path.join(path, '.git')):
        return path
    return get_git_root(os.path.dirname(path))


def get_git_commit(path):
    import git
    git_root = get_git_root(path)
    return git.Repo(git_root).head.commit.hexsha


def get_local_git_commit():
    import os
    return get_git_commit(os.getcwd())


import functools


@functools.cache
def get_df_from_wandb(path):
    import tqdm
    import wandb
    import pandas as pd

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(path)

    data_df_lines = []
    for run in tqdm.tqdm(runs):
        data_df_lines.append({
            'Name': run.name,
            'Commit': run.commit,
            **run.summary._json_dict,
            **{k: v for k, v in run.config.items() if not k.startswith('_')},
            'Config': run.config,
        })
    data_df = pd.DataFrame(data_df_lines)
    return data_df


def get_old_run(args):
    import os
    if os.environ.get('WANDB_NO_LOAD', '0') == '1':
        return None
    my_config = vars(args)
    old_runs = get_df_from_wandb(f'{os.environ["WANDB_ENTITY"]}/{os.environ["WANDB_PROJECT"]}')
    old_runs = old_runs[old_runs['Config'] == my_config].copy()
    # old_runs = old_runs[old_runs['Commit'] == get_local_git_commit()]
    if len(old_runs) == 0:
        return None

    if 'ppl_delta_by_layer_name_in_progress' not in old_runs.columns:
        old_runs['ppl_delta_by_layer_name_in_progress'] = None
    old_runs['ppl_delta_by_layer_name_in_progress_len'] = old_runs['ppl_delta_by_layer_name_in_progress'].apply(
        lambda ppl_delta_by_layer_name_in_progress: 0 if not isinstance(ppl_delta_by_layer_name_in_progress, dict)
        else len(ppl_delta_by_layer_name_in_progress)
    )
    return dict(old_runs.sort_values('ppl_delta_by_layer_name_in_progress_len', ascending=False).head(1).iloc[0])


###


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--edenn-d', type=int, required=True,
        help='EDENN grid dimension'
    )
    parser.add_argument(
        '--edenn-n', type=int, required=True,
        help='EDENN grid size'
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
        '--method', type=str, choices=["rtn", "gptq"], default="gptq", help="Method to quantize with",
    )
    parser.add_argument(
        '--dataset', type=str, default='red', choices=['red'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--div_loss', action='store_true',
        help='calculate KL divergence.'
    )
    parser.add_argument(
        '--random_init', action='store_true',
        help='initialize model with random weights.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=256,
        help='Number of calibration data samples.'
    )
    args = parser.parse_args()

    old_run = get_old_run(args)

    wandb.init(
        # track hyperparameters and run metadata
        config=args,
    )

    torch.set_grad_enabled(False)

    if args.random_init:
        def monkeypatch_torch_init():
            import torch
            import torch.nn as nn
            import transformers
            transformers.modeling_utils._init_weights = False
            TORCH_INIT_FUNCTIONS = {
                "uniform_": nn.init.uniform_,
                "normal_": nn.init.normal_,
                "trunc_normal_": nn.init.trunc_normal_,
                "constant_": nn.init.constant_,
                "xavier_uniform_": nn.init.xavier_uniform_,
                "xavier_normal_": nn.init.xavier_normal_,
                "kaiming_uniform_": nn.init.kaiming_uniform_,
                "kaiming_normal_": nn.init.kaiming_normal_,
                "uniform": nn.init.uniform,
                "normal": nn.init.normal,
                "xavier_uniform": nn.init.xavier_uniform,
                "xavier_normal": nn.init.xavier_normal,
                "kaiming_uniform": nn.init.kaiming_uniform,
                "kaiming_normal": nn.init.kaiming_normal,
            }
            for name in TORCH_INIT_FUNCTIONS.keys():
                def init_zeros(param, *args, **kwargs):
                    param.data = torch.zeros(param.shape, dtype=torch.float16)

                setattr(torch.nn.init, name, init_zeros)
        import transformers

        def get_random_init_model(model_name):
            monkeypatch_torch_init()
            config = transformers.AutoConfig.from_pretrained(model_name)
            model = transformers.AutoModelForCausalLM.from_config(config)

            # import tqdm
            # for param in tqdm.tqdm(list(model.parameters())):
            #     param.requires_grad = False
            #     param.data.view(-1)[0::2] = 0.1
            #     param.data.view(-1)[1::2] = -0.1
            #     param.data.view(-1)[0::3] = 0.1
            #     param.data.view(-1)[0::4] = -0.1
            return model.eval().half()
        model = get_random_init_model(args.model)
        import collections
        print(collections.Counter([param.dtype for param in model.parameters()]))
        print(collections.Counter([param.dtype for param in model.buffers()]))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )

    model.seqlen = args.seqlen
    model.eval()

    if (args.edenn_d, args.edenn_n) != (-1, -1):
        mse, _entropy = eval_grid(args.edenn_d, args.edenn_n)
        wandb.log({'test_grid_mse': mse})

    layers = sorted([
        layer for
        layer in find_layers(model).keys()
        if 'lm_head' not in layer
    ])

    baseline_ppl = None

    if old_run is not None:
        print('Skipping baseline')
        baseline_ppl = old_run.get('baseline_ppl', None)

    import math
    if not (isinstance(baseline_ppl, float) or isinstance(baseline_ppl, int)) or not math.isfinite(baseline_ppl):
        baseline_ppl = eval_ppl_by_config(args, model, get_empty_config(layers))

    wandb.log({'baseline_ppl': baseline_ppl}, commit=True)
    print(f'baseline_ppl: {baseline_ppl}')

    ppl_delta_by_layer_name = {}

    old_ppl_delta_by_layer_name_in_progress = {}
    if old_run is not None:
        old_ppl_delta_by_layer_name_in_progress = old_run.get('ppl_delta_by_layer_name_in_progress', {})

    print(old_ppl_delta_by_layer_name_in_progress)

    if not isinstance(old_ppl_delta_by_layer_name_in_progress, dict):
        old_ppl_delta_by_layer_name_in_progress = {}

    print(old_ppl_delta_by_layer_name_in_progress)

    for layer_idx, layer_name in enumerate(layers):
        print(f'Checking {layer_name}')
        if layer_name in old_ppl_delta_by_layer_name_in_progress:
            print(f'Skipping {layer_name}')
            ppl_delta = old_ppl_delta_by_layer_name_in_progress[layer_name]
        else:
            config = get_empty_config(layers)
            config[layer_name] = (args.edenn_d, args.edenn_n)

            ppl_delta = eval_ppl_by_config(
                args,
                model,
                config,
            ) - baseline_ppl

        wandb.log({'ppl_delta_by_layer_name_in_progress': ppl_delta_by_layer_name}, commit=True)
        ppl_delta_by_layer_name[layer_name] = ppl_delta
        print(f'ppl_delta: {ppl_delta}')

    wandb.log({'ppl_delta_by_layer_name': ppl_delta_by_layer_name}, commit=True)
    print(f'ppl_delta_by_layer_name: {ppl_delta_by_layer_name}')

    # model = model.to(DEV)
    # wandb.log(get_zero_shots(model, task_list = ['winogrande','piqa','hellaswag', 'arc_easy','arc_challenge'], num_fewshots=1))
    # wandb.log(get_zero_shots(model, task_list = ['mmlu',], num_fewshots=5))


if __name__ == '__main__':
    main()
