from typing import Union, Optional

import torch
from torch import nn

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        use_batch_norm: Optional[bool] = False,
        use_layer_norm: Optional[bool] = False,
        dropout_rate: Optional[float] = 0.0,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = [input_size] + [size for _ in range(n_layers)] + [output_size]
    num_layers = len(layers)

    network = nn.Sequential()
    for idx, (in_features, out_features) in enumerate(zip(layers[:-1], layers[1:])):
        network.add_module(module=nn.Linear(in_features, out_features), name=f'linear-{idx}')

        if use_batch_norm:
            network.add_module(module=nn.BatchNorm1d(out_features), name=f'batch_norm-{idx}')
        elif use_layer_norm:
            network.add_module(module=nn.LayerNorm(out_features), name=f'batch_norm-{idx}')

        if idx < num_layers - 2:
            network.add_module(module=activation, name=f'{activation}-{idx}')
        elif idx == num_layers - 1:
            network.add_module(module=output_activation, name=f'{activation}-{idx}')

        if dropout_rate > 0.0:
            network.add_module(module=nn.Dropout(p=dropout_rate), name=f'dropout-{idx}')

    return network

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    # raise NotImplementedError


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
