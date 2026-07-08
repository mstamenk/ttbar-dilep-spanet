#!/usr/bin/env python3
"""Quick dependency smoke test for the ROOT conversion environment."""

from __future__ import annotations

import importlib


PACKAGES = [
    "awkward",
    "h5py",
    "matplotlib",
    "numpy",
    "pandas",
    "scipy",
    "seaborn",
    "sklearn",
    "tqdm",
    "uproot",
    "vector",
    "yaml",
]


def main() -> None:
    for package in PACKAGES:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown")
        print(f"{package}: {version}")


if __name__ == "__main__":
    main()
