import torch.nn as nn


class MLP(nn.Module):
    """Fully-connected MLP with configurable depth and width.

    Args:
        hidden_sizes: list of ints, one per hidden layer.
        activation: activation class applied after each hidden layer.
    """

    def __init__(self, hidden_sizes: list[int], activation: type = nn.ReLU):
        super().__init__()
        dims = [2] + list(hidden_sizes) + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
