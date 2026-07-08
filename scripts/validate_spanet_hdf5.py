#!/usr/bin/env python3
"""Validate that an HDF5 file can be read by SPANET's native dataset loader."""

from __future__ import annotations

import argparse

from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Native SPANET HDF5 file.")
    parser.add_argument("--event-info", default="configs/ttbar_dilep_event.yaml", help="SPANET event YAML.")
    parser.add_argument("--limit", type=float, default=1.0, help="Dataset fraction to load.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = JetReconstructionDataset(
        args.input,
        args.event_info,
        limit_index=args.limit,
        partial_events=True,
    )
    print(f"events: {len(dataset)}")
    print(f"inputs: {dataset.event_info.input_names}")
    print(f"assignments: {list(dataset.assignments.keys())}")
    print(f"regressions: {list(dataset.regressions.keys())}")
    print(f"classifications: {list(dataset.classifications.keys())}")

    first = dataset[0]
    print(f"first event num_vectors: {first.num_vectors.item()}")
    print(f"first event assignment masks: {[target.mask.item() for target in first.assignment_targets]}")
    if "EVENT/reco_quality" in first.classification_targets:
        print(f"first event reco_quality: {first.classification_targets['EVENT/reco_quality'].item()}")


if __name__ == "__main__":
    main()
