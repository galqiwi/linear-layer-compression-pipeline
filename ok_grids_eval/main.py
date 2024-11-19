from edenn import higgs_quantize_dequantize
import torch
import os


DEV = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@torch.no_grad()
def eval_grid(edenn_d: int, edenn_n: int):
    x = torch.empty((2 ** 16, edenn_d), device=DEV).normal_()
    dequant, entropy = higgs_quantize_dequantize(x, edenn_d, edenn_n)
    mse = (x - dequant).pow(2).mean().item()
    return mse, entropy / edenn_d


def main():
    prefix = 'EDEN'
    suffix = '.pt'

    grids = [
        tuple(map(int, filename[len(prefix):-len(suffix)].split('-')))
        for filename in os.listdir('../grids')
        if (
            filename.endswith(suffix) and filename.startswith(prefix)
        )
    ]
    print(f'{grids=}')

    import math

    grids_config = [
        {
            'mse': eval_grid(edenn_d=edenn_d, edenn_n=edenn_n)[0],
            'bits': 16.0 / 1024.0 + math.log2(edenn_n) / edenn_d,
            'edenn_d': edenn_d,
            'edenn_n': edenn_n,
        }
        for edenn_d, edenn_n in grids
    ]
    grids_config.sort(key=lambda x: x['bits'])

    import pprint
    pprint.pprint(grids_config)


if __name__ == '__main__':
    main()

