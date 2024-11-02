import numpy as np
import pandas as pd
import tqdm
import wandb
import functools
import json
import requests
from ast import literal_eval
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from transformers import AutoModelForCausalLM


@functools.cache
def get_df_from_wandb(path):
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
        })
    data_df = pd.DataFrame(data_df_lines)
    return data_df


def find_grids_with_budget(
        slopes,  # linear coefficients for [layerwise mse -> metric]
        weights,  # linear coefficients for [layer bitwidth -> total bitwidth] (1 / num_blocks for blockwise)
        budget,  # upper bound on total bitwidth
        grid_bits,  # available grid bitwidths
        grid_mses  # available grid mses
) -> tuple[float, list]:
    num_layers = len(slopes)
    num_grids = len(grid_bits)
    assert len(grid_mses) == num_grids

    solver = pywraplp.Solver.CreateSolver("CP-SAT")

    x = {(j, i): solver.BoolVar("name") for i in range(num_grids) for j in range(num_layers)}

    for j in range(num_layers): solver.Add(sum(x[(j, i)] for i in range(num_grids)) == 1)
    solver.Add(
        sum(x[(j, i)] * weights[j] * grid_bits[i] for j in range(num_layers) for i in range(num_grids)) <= budget)
    solver.Minimize(sum(x[(j, i)] * slopes[j] * grid_mses[i] for j in range(num_layers) for i in range(num_grids)))

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        avg_bits = sum(
            x[(j, i)].solution_value() * weights[j] * grid_bits[i] for j in range(num_layers) for i in range(num_grids))
        solution = np.asarray([[x[(j, i)].solution_value() for i in range(num_grids)] for j in range(num_layers)])
        indices = np.argwhere(solution == 1.0)
        assert len(indices) == num_layers
        return avg_bits, indices[:, 1]
    else:
        raise Exception("Didn't solve")


def get_grids():
    grids = literal_eval(requests.get(
        'https://gist.githubusercontent.com/galqiwi/64533350e3dcf4dfa31cd33d9956efb4/raw/fbec4fe8eaf81ce0b946eae5ef2137b3a8cd0916/UPDATED_GRIDS_MSE'
    ).text)

    grids = pd.DataFrame(grids)
    grids['name'] = grids.apply(
        lambda row: 'edenn_d=' + str(row['edenn_d']) + ';edenn_n=' + str(row['edenn_n']),
        axis=1,
    )
    grids = grids[['bits', 'mse', 'name', 'edenn_d', 'edenn_n']]
    return grids


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


def get_config(args):
    slopes_df = get_df_from_wandb(args.slopes_wandb_name)

    slopes_df = slopes_df[['test_grid_mse', 'baseline_ppl', 'tag', 'ppl_delta_by_layer_name']]
    slopes_df = slopes_df[slopes_df['tag'] == args.tag]
    slopes_df = slopes_df.dropna()
    slopes_df = slopes_df.copy()

    baseline_ppl = slopes_df['baseline_ppl'].mean()

    ppl_delta_lines = []

    for _, row in slopes_df.iterrows():
        test_grid_mse = row['test_grid_mse']
        ppl_delta_by_layer_name = row['ppl_delta_by_layer_name']
        for layer, ppl_delta in ppl_delta_by_layer_name.items():
            ppl_delta_lines.append({
                'test_grid_mse': test_grid_mse,
                'layer': layer,
                'ppl_delta': ppl_delta,
            })

    layers = sorted(set(line['layer'] for line in ppl_delta_lines))

    for layer in layers:
        ppl_delta_lines.append({
            'test_grid_mse': 0.0,
            'layer': layer,
            'ppl_delta': 0.0,
        })

    ppl_delta_df = pd.DataFrame(ppl_delta_lines)

    layers = sorted(set(ppl_delta_df['layer']))

    slope_by_layer = {}

    for layer_idx, layer in enumerate(layers):
        to_fit = ppl_delta_df[ppl_delta_df['layer'] == layer]

        slope = LinearRegression(fit_intercept=False).fit(to_fit['test_grid_mse'].values.reshape(-1, 1),
                                                          (to_fit['ppl_delta']).values).coef_.item()

        slope_by_layer[layer] = slope

    ok_grids = get_grids()
    ok_grids = ok_grids[ok_grids['mse'] <= 4 ** -args.bits_sep]

    model_pt = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True, torch_dtype="auto", device_map='meta',
    )

    @functools.cache
    def get_numel(path):
        return get_module_by_path(model_pt, path).weight.numel()

    # solution

    layers = sorted(layers)

    scales = [slope_by_layer[layer] for layer in layers]
    numels = [get_numel(layer) for layer in layers]
    grid_edenn_d = ok_grids['edenn_d'].values
    grid_edenn_n = ok_grids['edenn_n'].values
    grid_bits = ok_grids['bits'].values
    grid_mses = ok_grids['mse'].values

    solution_size, solution_idxs = find_grids_with_budget(
        scales,
        numels,
        budget=sum(numels) * args.target_bits,
        grid_bits=grid_bits,
        grid_mses=grid_mses,
    )

    # output

    real_bits = solution_size / sum(numels)
    predicted_ppl = baseline_ppl + sum(
        max(grid_mses[solution_idx] * scale, 0.0) for solution_idx, scale in zip(solution_idxs, scales))
    optimal_config = {
        layer: (grid_edenn_d[solution_idx], grid_edenn_n[solution_idx])
        for layer, solution_idx in zip(layers, solution_idxs)
    }

    return {
        'real_bits': real_bits,
        'predicted_ppl': predicted_ppl,
        'optimal_config': optimal_config,
    }
