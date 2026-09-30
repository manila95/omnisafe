"""Persist per-eval-epoch value-study data (raw per-probe arrays + aggregate stats) to disk for
later offline analysis, and render a quick-look scatter-plot grid alongside it.

Complements the online logging (``progress.csv``/wandb/tensorboard, which only ever see the
aggregate stats -- ``MCStudy/*``, ``IntermediateMC/pos*/*``): those numbers are enough to watch
trends live, but reconstructing the underlying per-probe (predicted, MC-true) pairs after the fact
-- e.g. to look for outlier probes, refit a different correlation estimator, or build a custom
plot -- requires the raw arrays, which ``estimate_true_value_same_state_mc`` and
``estimate_value_from_snapshots`` only produce with ``return_raw=True`` and never persist
themselves.
"""

from __future__ import annotations

import os
import pickle


def save_eval_data(log_dir: str, epoch: int, bundle: dict, subdir: str = 'eval_data') -> str:
    """Pickle ``bundle`` to ``<log_dir>/<subdir>/epoch_{epoch:05d}.pkl``.

    Args:
        log_dir: The run's log directory (e.g. ``self._logger.log_dir``).
        epoch: Current epoch, used for both the filename and included in ``bundle`` for
            self-description (so a lone pickle file, moved elsewhere, still identifies itself).
        bundle: Arbitrary picklable data -- expected shape is a dict with keys like
            ``mc_study: {'stats': ..., 'raw': ...}`` and
            ``intermediate_study: {pos: {'stats': ..., 'raw': ...}, ...}``, but this function
            doesn't inspect or require any particular structure.
        subdir: Which subdirectory under ``log_dir`` to write to -- ``'eval_data'`` (default) for
            the MC-value-study bundle, ``'scatter_data'`` for the raw arrays behind
            ``Logger.log_scatter_image`` calls (see ``Logger.pop_scatter_raw_data``). Kept as one
            function rather than two near-duplicates since the pickling logic is identical either
            way; only the destination and the bundle's shape differ.

    Returns:
        The path written to.
    """
    out_dir = os.path.join(log_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'epoch_{epoch:05d}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    return path


def log_eval_data_to_wandb(
    pkl_path: str,
    epoch: int,
    name_prefix: str = 'eval-data',
    artifact_type: str = 'eval_data',
    description: str | None = None,
) -> None:
    """Sync a per-epoch data pickle (see :func:`save_eval_data`) to the active wandb run, so the
    raw arrays are downloadable from the run page without SSH access to whatever machine produced
    them -- previously only the aggregate stats (via ``Logger.store``/``dump_tabular``) and quick-
    look images (:func:`log_scatter_to_wandb`, ``Logger.log_scatter_image``) ever left local disk;
    the raw pickle itself was local-only.

    Uses a per-epoch :class:`wandb.Artifact` (one version per epoch) rather than ``wandb.save`` --
    an Artifact gets its own content-addressed version history and survives independently of the
    run's live file sync, so a pickle from epoch 100 stays fetchable (``wandb.Api().artifact(...)``)
    even long after the run itself has finished or if the plain run-files view gets pruned. The
    per-epoch granularity mirrors ``save_eval_data``'s own filename scheme (``epoch_{epoch:05d}.pkl``)
    precisely so a downloaded artifact identifies itself the same way the local copy does.

    Args:
        pkl_path: Path returned by :func:`save_eval_data`.
        epoch: Current epoch -- used for both the artifact's name suffix and the wandb step, so
            this lands on the same x-axis position as that epoch's numeric eval metrics.
        name_prefix: Artifact name becomes ``f'{name_prefix}-epoch-{epoch:05d}'`` -- override to
            distinguish multiple data streams (e.g. ``'scatter-data'`` for the
            ``Logger.pop_scatter_raw_data`` bundle) sharing this same upload path.
        artifact_type: wandb Artifact ``type``, distinct per data stream for the same reason.
        description: Optional artifact description; a generic one is used if omitted.
    """
    import wandb  # noqa: PLC0415 -- see save_scatter_grid's matplotlib import for why this is local

    if wandb.run is None:
        return
    artifact = wandb.Artifact(
        name=f'{name_prefix}-epoch-{epoch:05d}',
        type=artifact_type,
        description=description or f'Raw per-probe arrays + aggregate stats, epoch {epoch}.',
    )
    artifact.add_file(pkl_path, name=os.path.basename(pkl_path))
    wandb.log_artifact(artifact)
