#!/usr/bin/env python3
"""Produce quick tables and plots for a ttbar SPANET HDF5 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input HDF5 file from scripts/root_to_hdf5.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for plots and summary JSON.")
    return parser.parse_args()


def counts(values: np.ndarray) -> dict[str, int]:
    unique, count = np.unique(values, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, count, strict=True)}


def save_hist(path: Path, values: np.ndarray, title: str, xlabel: str, weights: np.ndarray | None = None, bins: int = 50) -> None:
    plt.figure(figsize=(7, 5))
    sns.histplot(x=values, weights=weights, bins=bins, element="step", fill=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Events")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as h5:
        jets = h5["inputs/jets"][:]
        leptons = h5["inputs/leptons"][:]
        met = h5["inputs/met"][:]
        pair_label = h5["targets/pair_label"][:]
        reco_quality = h5["targets/reco_quality"][:]
        pair_mask = h5["masks/pair"][:].astype(bool)
        reco_mask = h5["masks/reco"][:].astype(bool) if "reco" in h5["masks"] else np.ones_like(pair_mask, dtype=bool)
        nu_mask = h5["masks/nu"][:].astype(bool)
        top_mask = h5["masks/top"][:].astype(bool)
        split = h5["metadata/split"][:]
        event = h5["metadata/event"][:]
        weights = h5["weights/event"][:]
        attrs = {key: h5.attrs[key] for key in h5.attrs}

    duplicate_events = int(len(event) - len(np.unique(event)))
    split_pair_counts = {}
    for split_code, split_name in ((0, "train"), (1, "val"), (2, "test")):
        selected = split == split_code
        valid_pair = selected & pair_mask
        split_pair_counts[split_name] = {
            "events": int(np.sum(selected)),
            "pair_valid": int(np.sum(valid_pair)),
            "pair_label_counts": counts(pair_label[valid_pair]) if np.any(valid_pair) else {},
            "weighted_yield": float(np.sum(weights[selected])),
        }

    summary = {
        "input": args.input,
        "events": int(len(split)),
        "duplicate_event_ids": duplicate_events,
        "split_counts": counts(split),
        "reco_quality_counts": counts(reco_quality),
        "reco_quality_counts_valid_only": counts(reco_quality[reco_mask]) if np.any(reco_mask) else {},
        "reco_valid": int(np.sum(reco_mask)),
        "pair_label_counts_valid_only": counts(pair_label[pair_mask]) if np.any(pair_mask) else {},
        "nu_mask_counts": {
            "nu": int(np.sum(nu_mask[:, 0])),
            "nubar": int(np.sum(nu_mask[:, 1])),
        },
        "top_mask_counts": {
            "top": int(np.sum(top_mask[:, 0])),
            "tbar": int(np.sum(top_mask[:, 1])),
        },
        "weighted_yield_total": float(np.sum(weights)),
        "split_pair_counts": split_pair_counts,
        "attrs": attrs,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, default=str)

    save_hist(output_dir / "jet_pt.png", jets[:, :, 0].reshape(-1), "Selected jet pT", "pT [GeV]")
    save_hist(output_dir / "electron_pt.png", leptons[:, 0, 0], "Electron pT", "pT [GeV]", weights=weights)
    save_hist(output_dir / "muon_pt.png", leptons[:, 1, 0], "Muon pT", "pT [GeV]", weights=weights)
    save_hist(output_dir / "met_pt.png", met[:, 0], "MET pT", "pT [GeV]", weights=weights)

    plt.figure(figsize=(6, 5))
    sns.countplot(x=split)
    plt.xlabel("split code")
    plt.ylabel("events")
    plt.tight_layout()
    plt.savefig(output_dir / "split_counts.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.countplot(x=reco_quality)
    plt.xlabel("reco quality")
    plt.ylabel("events")
    plt.tight_layout()
    plt.savefig(output_dir / "reco_quality_counts.png")
    plt.close()

    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
