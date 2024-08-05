import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QLinear(nn.Module):
    """
    Linear transformation with quantization support [1]. Implementation was
    stolen from geohot's notebooks [2].

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
        Size of each output sample.
    bias: bool
        If set to ``False``, the layer will not learn an additive bias.

    References
    ----------
    .. [1] Self-Compressing Neural Networks. (https://arxiv.org/pdf/2301.13142)
    .. [2] https://github.com/geohot/ai-notebooks/blob/master/mnist_self_compression.ipynb
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
    e: torch.Tensor
    b: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.e = nn.parameter.Parameter(
            torch.full((out_features, 1), -4.0, **factory_kwargs)
        )
        self.b = nn.parameter.Parameter(
            torch.full((out_features, 1), 3.0, **factory_kwargs)
        )
        if bias:
            self.bias = nn.parameter.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qw = self.qweight
        w = (qw.round() - qw).detach() + qw  # straight through estimator
        w = 2 ** self.e * w
        return nn.functional.linear(
            input, w, self.bias * self.b.relu().squeeze(-1)
        )

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}), '
            + f'out_features={self.out_features}, '
            + f' bias={self.bias is not None}'
        )

    def strip(self, indices: torch.Tensor, dim: int) -> 'QLinear':
        if dim not in [0, 1]:
            raise IndexError('dim should be 0 or 1!')

        # strip in channels
        if dim == 1:
            result = QLinear(
                len(indices),
                self.out_features,
                self.bias is not None,
                self.weight.device,
                self.weight.dtype
            )
            with torch.no_grad():
                result.weight.copy_(self.weight[:, indices])
                if self.bias is not None:
                    result.bias.copy_(self.bias)
                result.e.copy_(self.e)
                result.b.copy_(self.b)

        # strip out channels
        if dim == 0:
            result = QLinear(
                self.in_features,
                len(indices),
                self.bias is not None,
                self.weight.device,
                self.weight.dtype
            )
            with torch.no_grad():
                result.weight.copy_(self.weight[indices, :])
                if self.bias is not None:
                    result.bias.copy_(self.bias[indices])
                result.e.copy_(self.e[indices])
                result.b.copy_(self.b[indices])

        result = result.train(self.training)

        return result

    def freeze(self) -> nn.Linear:
        result = nn.Linear(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.weight.device,
            self.weight.dtype
        )

        with torch.no_grad():
            qw = self.qweight
            w = (qw.round() - qw).detach() + qw
            w = 2 ** self.e * w
            result.weight.copy_(w)
            if self.bias is not None:
                result.bias.copy_(self.bias * self.b.relu().squeeze(-1))

        result = result.train(self.training)

        return result

    @property
    def n_weights_alive(self) -> torch.Tensor:
        return len(torch.nonzero(self.b.relu())) * self.weight.shape[1]

    @property
    def alive_weight_indices(self) -> torch.Tensor:
        return torch.nonzero(self.b.relu())[:, 0]

    @property
    def qbits(self) -> torch.Tensor:
        return self.b.relu().sum() * self.weight.shape[1]

    @property
    def qweight(self) -> torch.Tensor:
        b_plus = self.b.relu() - 1
        return torch.min(
            torch.max(2 ** -self.e * self.weight, -2 ** b_plus),
            2 ** b_plus - 1
        )


class Shifted_Softplus(torch.nn.Softplus):
    """Element-wise softplus function shifted as to pass from the origin."""

    def __init__(self, beta=1, threshold=20):
        super(Shifted_Softplus, self).__init__(beta, threshold)

    def forward(self, input):
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0


class Custom_Sigmoid(torch.nn.Module):
    def __init__(self, p=3):
        super(Custom_Sigmoid, self).__init__()
        self.p = p

    def forward(self, input):
        return 1 / (1 + torch.exp(-self.p*(input)))


def get_activation(activation: str):
    """Return activation module given string."""
    activ = None
    if activation == "relu":
        activ = torch.nn.ReLU(True)
    elif activation == "elu":
        activ = torch.nn.ELU(True)
    elif activation == "tanh":
        activ = torch.nn.Tanh()
    elif activation == "softplus":
        activ = torch.nn.Softplus()
    elif activation == "shifted_softplus":
        activ = Shifted_Softplus()
    elif activation == "custom_sigmoid":
        activ = Custom_Sigmoid()
    elif activation == "linear":
        print("WARNING: no activation selected")
    elif activation is None:
        pass
    else:
        raise ValueError(
            f"Unknown activation: {activation}. options: 'relu','elu','tanh','softplus','shifted_softplus','linear'. "
        )
    return activ


def parse_nn_options(options: str, n_layers: int, last_layer_activation: bool):
    """Parse args per layer of the NN.

    If a single value is given, repeat options to all layers but for the output one,
    unless ``last_layer_activation is True``, in which case the option is repeated
    also for the output layer.
    """
    # If an iterable is given cheeck that its length matches the number of NN layers
    if hasattr(options, "__iter__") and not isinstance(options, str):
        if len(options) != n_layers:
            raise ValueError(
                f"Length of options: {options} ({len(options)} should be equal to number of layers ({n_layers}))."
            )
        options_list = options
    # if a single value is given, repeat options to all layers but for the output one
    else:
        if last_layer_activation:
            options_list = [options for _ in range(n_layers)]
        else:
            options_list = [options for _ in range(n_layers - 1)]
            options_list.append(None)

    return options_list
