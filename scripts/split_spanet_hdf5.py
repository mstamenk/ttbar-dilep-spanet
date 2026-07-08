#!/usr/bin/env python3
"""Split a converted HDF5 file into train/val/test files using metadata/split."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


SPLITS = {"train": 0, "val": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Full converted HDF5 file.")
    parser.add_argument("--output-dir", required=True, help="Directory for split HDF5 files.")
    parser.add_argument("--prefix", default="ttbar_dilep_v1")
    return parser.parse_args()


def recursive_copy_subset(src_group: h5py.Group, dst_group: h5py.Group, indices: np.ndarray, n_events: int) -> None:
    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            recursive_copy_subset(item, dst_group.create_group(name), indices, n_events)
            continue

        data = item[:]
        if data.shape and data.shape[0] == n_events:
            data = data[indices]
        dst_group.create_dataset(name, data=data, compression="lzf")


def write_split(input_file: str, output_file: Path, split_name: str, split_code: int) -> int:
    with h5py.File(input_file, "r") as src:
        split = src["metadata/split"][:]
        indices = np.flatnonzero(split == split_code)
        n_events = len(split)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_file, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["source_file"] = input_file
            dst.attrs["source_split_name"] = split_name
            dst.attrs["source_split_code"] = split_code
            recursive_copy_subset(src, dst, indices, n_events)

    return len(indices)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for split_name, split_code in SPLITS.items():
        output_file = output_dir / f"{args.prefix}_{split_name}.h5"
        count = write_split(args.input, output_file, split_name, split_code)
        print(f"{split_name}: {count} events -> {output_file}")


if __name__ == "__main__":
    main()
