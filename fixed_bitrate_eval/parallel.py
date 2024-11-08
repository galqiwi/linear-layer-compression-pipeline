import torch
import torch.nn as nn


def replace_submodule(module, submodule_path, new_submodule):
    submodule_names = submodule_path.split(".")
    for submodule in submodule_names[:-1]:
        module = getattr(module, submodule)
    setattr(module, submodule_names[-1], new_submodule)


def get_submodule(module, submodule_path):
    submodule_names = submodule_path.split(".")
    for submodule in submodule_names:
        module = getattr(module, submodule)
    return module


class LinearModelParallel(nn.Module):
    def __init__(self, inner, devices):
        super().__init__()
        assert isinstance(inner, nn.Linear)
        assert inner.bias is None
        out_dim, in_dim = inner.weight.shape
        self.inner_layers = nn.ModuleList([])

        for device_idx, device in enumerate(devices):
            assert out_dim % len(devices) == 0
            out_chunk_size = out_dim // len(devices)

            new_layer = nn.Linear(in_features=in_dim, out_features=out_chunk_size, bias=False, device='meta')
            new_layer.weight = nn.Parameter(
                inner.weight.data[out_chunk_size * device_idx: out_chunk_size * (device_idx + 1)].detach().to(
                    device),
                requires_grad=False,
            )
            self.inner_layers.append(new_layer)

    def forward(self, input_):
        # print(f'Linear forward {input_.shape}')

        def inner_layer_forward(inner_layer, input_):
            return inner_layer(input_)

        funcs_by_replica = [inner_layer_forward for inner_layer in self.inner_layers]
        inputs_by_replica = [
            (inner_layer, input_.to(inner_layer.weight.device))
            for inner_layer in self.inner_layers
        ]
        devices = [
            inner_layer.weight.device
            for inner_layer in self.inner_layers
        ]
        outputs = torch.nn.parallel.parallel_apply(funcs_by_replica, inputs_by_replica, devices=devices)

        output = torch.concatenate([output.to(input_.device) for output in outputs], dim=-1)
        return output


class EmbedModelParallel(nn.Module):
    def __init__(self, inner, devices):
        super().__init__()
        assert isinstance(inner, nn.Embedding)
        num_embeddings, embedding_dim = inner.weight.shape
        self.inner_layers = nn.ModuleList([])

        for device_idx, device in enumerate(devices):
            assert embedding_dim % len(devices) == 0
            out_chunk_size = embedding_dim // len(devices)

            new_layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=out_chunk_size, device='meta')
            new_layer.weight = nn.Parameter(
                inner.weight.data[:,
                out_chunk_size * device_idx: out_chunk_size * (device_idx + 1)].detach().to(device),
                requires_grad=False,
            )
            self.inner_layers.append(new_layer)

    def forward(self, input_):
        # print(f'Embed forward {input_.shape}')

        def inner_layer_forward(inner_layer, input_):
            return inner_layer(input_)

        funcs_by_replica = [inner_layer_forward for inner_layer in self.inner_layers]
        inputs_by_replica = [
            (inner_layer, input_.to(inner_layer.weight.device))
            for inner_layer in self.inner_layers
        ]
        devices = [
            inner_layer.weight.device
            for inner_layer in self.inner_layers
        ]
        outputs = torch.nn.parallel.parallel_apply(funcs_by_replica, inputs_by_replica, devices=devices)

        output = torch.concatenate([output.to(input_.device) for output in outputs], dim=-1)
        return output


def dispatch_model_parallel(model, devices=None, verbose=True):
    if devices is None:
        devices = [
            torch.device(type='cuda', index=device_idx)
            for device_idx in range(torch.cuda.device_count())
        ]

    print(devices)

    if len(devices) == 1:
        return model.to(devices[0])

    import tqdm

    type_by_nuclear_module_name = {
        module_name: type(module)
        for module_name, module in model.named_modules()
        if len(list(module.named_modules())) == 1
    }

    iter_ = type_by_nuclear_module_name.items()
    iter_ = sorted(iter_)
    if verbose:
        iter_ = tqdm.tqdm(iter_, desc='Dispatching model')

    for module_name, module_type in iter_:
        if module_type == nn.Linear:
            replace_submodule(model, module_name, LinearModelParallel(get_submodule(model, module_name), devices))
        elif module_type == nn.Embedding:
            replace_submodule(model, module_name, EmbedModelParallel(get_submodule(model, module_name), devices))
        else:
            replace_submodule(model, module_name, get_submodule(model, module_name).to(devices[0]))

    return model
