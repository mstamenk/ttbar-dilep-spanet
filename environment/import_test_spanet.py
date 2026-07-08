#!/usr/bin/env python3
"""Quick dependency smoke test for the SPANET training environment."""

from __future__ import annotations

import importlib


PACKAGES = [
    "h5py",
    "matplotlib",
    "numba",
    "numpy",
    "pytorch_lightning",
    "seaborn",
    "sklearn",
    "spanet",
    "sympy",
    "torch",
    "tqdm",
    "yaml",
]


def main() -> None:
    for package in PACKAGES:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown")
        print(f"{package}: {version}")

    import site
    import torch

    print(f"user site enabled: {site.ENABLE_USER_SITE}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch cuda device: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
