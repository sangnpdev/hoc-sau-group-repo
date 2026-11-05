"""Mạng MLP tuỳ biến số tầng ẩn và số nơ-ron mỗi tầng."""
import torch.nn as nn

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}


class MLP(nn.Module):
    """2 đầu vào → [Linear → activation] × C → Linear → 1 đầu ra (tuyến tính, cho hồi quy)."""

    def __init__(self, hidden_sizes, activation="tanh", in_dim=2, out_dim=1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), ACTIVATIONS[activation]()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

