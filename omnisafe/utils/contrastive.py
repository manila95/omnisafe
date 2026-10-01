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
"""Time-contrastive pair sampling and InfoNCE loss for ``phi_source='contrastive'``.

States visited close together in time within one episode are pulled together in phi-space,
temporally distant or other-episode ones pushed apart. Plain tensor-in/tensor-out functions; the
optimizer loop that calls them lives in ``PolicyGradient._contrastive_update_phi``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_temporal_pairs(
    lengths: list[int],
    horizon: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""One (anchor, positive) row-index pair per valid row of a segmented batch.

    ``lengths`` describes concatenated episode segments, each contiguous and time-ordered. For
    every row ``t`` in a segment of length ``L``, draws one offset uniformly from
    ``{-min(horizon, t), ..., -1} u {1, ..., min(horizon, L-1-t)}`` -- within the same episode,
    never ``t`` itself. Rows with an empty window (only when ``L == 1``) are dropped.

    Args:
        lengths: Length of each segment, in the order its rows appear in the flat batch.
        horizon: Maximum ``|offset|`` a positive may be drawn at. Must be >= 1.
        generator: For reproducible sampling; must live on ``device``.
        device: Device to build the index tensors on -- pass the batch's own, since CPU indices
            into a CUDA tensor work but not the reverse.

    Returns:
        ``(anchor_idx, positive_idx)``, int64 tensors of equal shape indexing the flat batch.
    """
    assert horizon >= 1, f'horizon must be >= 1, got {horizon}.'
    if not lengths:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    n = int(lengths_t.sum().item())
    device = lengths_t.device
    seg_starts = torch.cumsum(lengths_t, dim=0) - lengths_t
    row_start = torch.repeat_interleave(seg_starts, lengths_t)
    row_length = torch.repeat_interleave(lengths_t, lengths_t)
    global_idx = torch.arange(n, device=device)
    t = global_idx - row_start
    n_neg = torch.clamp(torch.clamp(t, max=horizon), min=0)
    n_pos = torch.clamp(torch.clamp(row_length - 1 - t, max=horizon), min=0)
    total = n_neg + n_pos
    valid = total > 0
    if not bool(valid.any()):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    anchor_idx = global_idx[valid]
    n_neg, total = n_neg[valid], total[valid]
    u = (torch.rand(anchor_idx.shape[0], generator=generator, device=device) * total).floor().long()
    u = torch.clamp(u, max=total - 1)
    k = torch.where(u < n_neg, -(n_neg - u), 1 + (u - n_neg))
    return anchor_idx, anchor_idx + k


def info_nce_loss(
    anchor_feat: torch.Tensor,
    positive_feat: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""Symmetric InfoNCE / NT-Xent loss with in-batch negatives.

    Both inputs are assumed already l2-normalized, so ``anchor @ positive.T`` is a cosine
    similarity matrix whose diagonal holds the true pairs.

    Args:
        anchor_feat: ``(M, sr_dim)`` anchor features.
        positive_feat: ``(M, sr_dim)`` features of each anchor's positive.
        temperature: Divides the similarities before the cross-entropy. Lower is sharper.

    Returns:
        ``(loss, stats)`` -- stats are detached, un-scaled cosine similarities for logging.
    """
    assert anchor_feat.shape == positive_feat.shape, (
        f'anchor_feat {tuple(anchor_feat.shape)} and positive_feat {tuple(positive_feat.shape)} '
        'must have the same shape.'
    )
    m = anchor_feat.shape[0]
    sim = (anchor_feat @ positive_feat.T) / temperature
    labels = torch.arange(m, device=sim.device)
    loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))

    with torch.no_grad():
        diag = sim.diagonal()
        pos_sim = (diag.mean() * temperature).item()
        neg_sim = (
            ((sim.sum() - diag.sum()) / (m * (m - 1)) * temperature).item()
            if m > 1
            else float('nan')
        )
    return loss, {'Loss': loss.item(), 'PosSim': pos_sim, 'NegSim': neg_sim}
