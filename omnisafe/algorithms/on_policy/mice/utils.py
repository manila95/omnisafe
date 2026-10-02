"""Helpers for MICE."""

import torch


class RandomProjection(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RandomProjection, self).__init__()
        self.linear_projection = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear_projection(x)
