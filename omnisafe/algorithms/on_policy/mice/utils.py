# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helpers for MICE."""

from __future__ import annotations

import torch


class RandomProjection(torch.nn.Module):
    """Fixed random linear embedding, never trained.

    The flashbulb memory's k-NN lookup runs in this space rather than raw observation space.

    Args:
        input_dim: Observation dimension.
        output_dim: Embedding dimension (``model_cfgs.emb_dim``).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear_projection = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed ``x``."""
        return self.linear_projection(x)
