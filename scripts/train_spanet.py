#!/usr/bin/env python3
"""Launch SPANET training with compatibility shims for the installed stack."""

from __future__ import annotations

import importlib.util
import runpy
import sys
import types
import warnings


def patch_lightning_private_imports() -> None:
    """Patch old SPANET imports onto newer PyTorch Lightning layouts."""
    import pytorch_lightning as pl
    import pytorch_lightning.profilers as lightning_profilers
    import pytorch_lightning.strategies as lightning_strategies
    import pytorch_lightning.utilities.imports as lightning_imports

    if not hasattr(lightning_imports, "_RICH_AVAILABLE"):
        lightning_imports._RICH_AVAILABLE = importlib.util.find_spec("rich") is not None

    if "pytorch_lightning.profiler" not in sys.modules:
        profiler_module = types.ModuleType("pytorch_lightning.profiler")
        profiler_module.PyTorchProfiler = lightning_profilers.PyTorchProfiler
        sys.modules["pytorch_lightning.profiler"] = profiler_module

    if not hasattr(lightning_strategies, "DDPFullyShardedStrategy") and hasattr(lightning_strategies, "FSDPStrategy"):
        lightning_strategies.DDPFullyShardedStrategy = lightning_strategies.FSDPStrategy

    original_trainer = pl.Trainer

    class CompatibleTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            self._resume_from_checkpoint_compat = kwargs.pop("resume_from_checkpoint", None)
            kwargs.pop("track_grad_norm", None)
            if kwargs.get("accelerator") is None:
                kwargs.pop("accelerator")
            if kwargs.get("devices") is None:
                kwargs.pop("devices")
            super().__init__(*args, **kwargs)

        def fit(self, model, train_dataloaders=None, val_dataloaders=None, datamodule=None, ckpt_path=None):
            if ckpt_path is None:
                ckpt_path = self._resume_from_checkpoint_compat
            return super().fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path=ckpt_path)

    pl.Trainer = CompatibleTrainer


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="No device id is provided via `init_process_group` or `barrier `.*",
        category=UserWarning,
    )
    patch_lightning_private_imports()
    runpy.run_module("spanet.train", run_name="__main__")


if __name__ == "__main__":
    main()
