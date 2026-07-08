#!/usr/bin/env python3
"""Run SPANET inference on the source ROOT ntuple and write one augmented ROOT file."""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import torch
import uproot
from tqdm import tqdm

try:
    import packaging
    import setuptools

    if not hasattr(setuptools, "extern"):
        setuptools.extern = SimpleNamespace(packaging=packaging)
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from root_to_hdf5 import (  # noqa: E402
    DEFAULT_BRANCHES,
    FLOAT,
    INT,
    as_numpy,
    comma_list,
    delta_phi,
    delta_r,
    expand_inputs,
    invariant_mass,
    make_chunk,
    resolve_branches,
    scalar,
    selected_first,
    threadpool_limits,
    visible_cpu_count,
    worker_counts,
)
from spanet.dataset.types import Source  # noqa: E402
from spanet.evaluation import load_model  # noqa: E402
from spanet.network.jet_reconstruction.jet_reconstruction_network import extract_predictions  # noqa: E402


REGRESSION_KEYS = (
    "EVENT/nu_px",
    "EVENT/nu_py",
    "EVENT/nu_pz",
    "EVENT/nubar_px",
    "EVENT/nubar_py",
    "EVENT/nubar_pz",
    "EVENT/top_px",
    "EVENT/top_py",
    "EVENT/top_pz",
    "EVENT/top_e",
    "EVENT/tbar_px",
    "EVENT/tbar_py",
    "EVENT/tbar_pz",
    "EVENT/tbar_e",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Input ROOT files or glob patterns.")
    parser.add_argument("--output", required=True, help="Output augmented ROOT file.")
    parser.add_argument("--tree", default="Events", help="Input TTree name.")
    parser.add_argument("--output-tree", default="Events", help="Output TTree name.")
    parser.add_argument("--log-dir", required=True, help="SPANET run version directory.")
    parser.add_argument("--reference-h5", required=True, help="SPANET HDF5 file used only to instantiate the model metadata.")
    parser.add_argument("--event-info", default="configs/ttbar_dilep_event.yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to latest epoch checkpoint.")
    parser.add_argument("--batch-size", type=int, default=65536, help="Inference batch size.")
    parser.add_argument("--chunk-events", type=int, default=200000, help="Input ROOT events per chunk.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap on selected events.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Inference device.")
    parser.add_argument("--rank", type=int, default=0, help="Shard rank for multi-process inference.")
    parser.add_argument("--world-size", type=int, default=1, help="Number of ROOT entry-range shards.")
    parser.add_argument("--pair-dr-max", type=float, default=0.4, help="Truth matching dR, for copied truth labels only.")
    parser.add_argument("--workers", type=int, default=None, help="Total CPU workers. Defaults to all visible CPUs.")
    parser.add_argument("--decompression-workers", type=int, default=None)
    parser.add_argument("--interpretation-workers", type=int, default=None)
    parser.add_argument("--compute-workers", type=int, default=None)
    parser.add_argument(
        "--original-branches",
        choices=("all", "minimal", "none"),
        default="all",
        help="Original selected ROOT branches to copy into the output.",
    )

    for key, aliases in DEFAULT_BRANCHES.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}-branches",
            default=",".join(aliases),
            type=comma_list,
            help=argparse.SUPPRESS,
        )

    return parser.parse_args()


def selection_mask(events: ak.Array, branches: dict[str, str | None]) -> ak.Array:
    n_jet = ak.num(events[branches["jet_pt"]])
    n_ele = ak.num(events[branches["electron_pt"]])
    n_mu = ak.num(events[branches["muon_pt"]])
    ele_charge = selected_first(events[branches["electron_charge"]])
    mu_charge = selected_first(events[branches["muon_charge"]])
    return (n_jet == 2) & (n_ele == 1) & (n_mu == 1) & (ele_charge * mu_charge < 0)


def source_args(args: argparse.Namespace) -> SimpleNamespace:
    values = vars(args).copy()
    values.setdefault("val_fraction", 0.15)
    values.setdefault("test_fraction", 0.15)
    values.setdefault("compression", "lzf")
    return SimpleNamespace(**values)


def log1p_features(data: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    output = data.copy()
    for index in indices:
        output[..., index] = np.log(output[..., index] + 1.0)
    return output


def build_sources(chunk: dict[str, np.ndarray], device: torch.device) -> tuple[Source, ...]:
    jets = np.stack(
        [
            chunk["jets"][:, :, 0],
            chunk["jets"][:, :, 1],
            np.sin(chunk["jets"][:, :, 2]),
            np.cos(chunk["jets"][:, :, 2]),
            chunk["jets"][:, :, 3],
        ],
        axis=-1,
    ).astype(FLOAT)
    jets = log1p_features(jets, (0, 4))

    leptons = chunk["leptons"]
    lepton_features = np.stack(
        [
            leptons[:, 0, 0],
            leptons[:, 0, 1],
            np.sin(leptons[:, 0, 2]),
            np.cos(leptons[:, 0, 2]),
            leptons[:, 0, 3],
            leptons[:, 0, 4],
            leptons[:, 1, 0],
            leptons[:, 1, 1],
            np.sin(leptons[:, 1, 2]),
            np.cos(leptons[:, 1, 2]),
            leptons[:, 1, 3],
            leptons[:, 1, 4],
        ],
        axis=-1,
    ).astype(FLOAT)
    lepton_features = log1p_features(lepton_features[:, None, :], (0, 4, 6, 10))

    met = np.stack(
        [chunk["met"][:, 0], np.sin(chunk["met"][:, 1]), np.cos(chunk["met"][:, 1])],
        axis=-1,
    ).astype(FLOAT)
    met = log1p_features(met[:, None, :], (0,))

    event_features = log1p_features(chunk["event_features"][:, None, :], (5, 6, 7))

    n_events = chunk["jets"].shape[0]
    return (
        Source(torch.as_tensor(jets, device=device), torch.ones((n_events, 2), dtype=torch.bool, device=device)),
        Source(torch.as_tensor(lepton_features, device=device), torch.ones((n_events, 1), dtype=torch.bool, device=device)),
        Source(torch.as_tensor(met, device=device), torch.ones((n_events, 1), dtype=torch.bool, device=device)),
        Source(torch.as_tensor(event_features, device=device), torch.ones((n_events, 1), dtype=torch.bool, device=device)),
    )


def positive_energy(px: np.ndarray, py: np.ndarray, pz: np.ndarray, mass: float | np.ndarray = 0.0) -> np.ndarray:
    return np.sqrt(np.maximum(px * px + py * py + pz * pz + mass * mass, 0.0)).astype(FLOAT)


def mass_from_cartesian(px: np.ndarray, py: np.ndarray, pz: np.ndarray, energy: np.ndarray) -> np.ndarray:
    mass2 = energy * energy - px * px - py * py - pz * pz
    return np.sqrt(np.maximum(mass2, 0.0)).astype(FLOAT)


def pxpypze_from_pt_eta_phi_mass(pt, eta, phi, mass) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = positive_energy(px, py, pz, mass)
    return px.astype(FLOAT), py.astype(FLOAT), pz.astype(FLOAT), energy


def combine_mass(vectors: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]) -> np.ndarray:
    px = sum(vector[0] for vector in vectors)
    py = sum(vector[1] for vector in vectors)
    pz = sum(vector[2] for vector in vectors)
    energy = sum(vector[3] for vector in vectors)
    return mass_from_cartesian(px, py, pz, energy)


def take_by_index(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return values[np.arange(values.shape[0]), indices]


def derived_masses(chunk: dict[str, np.ndarray], arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    leptons = chunk["leptons"]
    jets = chunk["jets"]
    e_charge = leptons[:, 0, 4]

    e_vec = pxpypze_from_pt_eta_phi_mass(leptons[:, 0, 0], leptons[:, 0, 1], leptons[:, 0, 2], leptons[:, 0, 3])
    mu_vec = pxpypze_from_pt_eta_phi_mass(leptons[:, 1, 0], leptons[:, 1, 1], leptons[:, 1, 2], leptons[:, 1, 3])

    top_e_index = arrays["spanet_topE_b_index"]
    top_mu_index = arrays["spanet_topMu_b_index"]
    top_e_jet = pxpypze_from_pt_eta_phi_mass(
        take_by_index(jets[:, :, 0], top_e_index),
        take_by_index(jets[:, :, 1], top_e_index),
        take_by_index(jets[:, :, 2], top_e_index),
        take_by_index(jets[:, :, 3], top_e_index),
    )
    top_mu_jet = pxpypze_from_pt_eta_phi_mass(
        take_by_index(jets[:, :, 0], top_mu_index),
        take_by_index(jets[:, :, 1], top_mu_index),
        take_by_index(jets[:, :, 2], top_mu_index),
        take_by_index(jets[:, :, 3], top_mu_index),
    )

    nu = (
        arrays["spanet_nu_px"],
        arrays["spanet_nu_py"],
        arrays["spanet_nu_pz"],
        positive_energy(arrays["spanet_nu_px"], arrays["spanet_nu_py"], arrays["spanet_nu_pz"]),
    )
    nubar = (
        arrays["spanet_nubar_px"],
        arrays["spanet_nubar_py"],
        arrays["spanet_nubar_pz"],
        positive_energy(arrays["spanet_nubar_px"], arrays["spanet_nubar_py"], arrays["spanet_nubar_pz"]),
    )

    electron_uses_nu = e_charge > 0
    e_nu = tuple(np.where(electron_uses_nu, nu[i], nubar[i]).astype(FLOAT) for i in range(4))
    mu_nu = tuple(np.where(electron_uses_nu, nubar[i], nu[i]).astype(FLOAT) for i in range(4))

    top_mass = mass_from_cartesian(
        arrays["spanet_top_px"], arrays["spanet_top_py"], arrays["spanet_top_pz"], arrays["spanet_top_e"]
    )
    tbar_mass = mass_from_cartesian(
        arrays["spanet_tbar_px"], arrays["spanet_tbar_py"], arrays["spanet_tbar_pz"], arrays["spanet_tbar_e"]
    )

    return {
        "spanet_w_e_mass": combine_mass((e_vec, e_nu)),
        "spanet_w_mu_mass": combine_mass((mu_vec, mu_nu)),
        "spanet_topE_lnu_b_mass": combine_mass((e_vec, e_nu, top_e_jet)),
        "spanet_topMu_lnu_b_mass": combine_mass((mu_vec, mu_nu, top_mu_jet)),
        "spanet_top_regressed_mass": top_mass,
        "spanet_tbar_regressed_mass": tbar_mass,
    }


def predict_chunk(model, chunk: dict[str, np.ndarray], device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    n_events = chunk["jets"].shape[0]
    outputs: dict[str, list[np.ndarray]] = {}

    for start in range(0, n_events, batch_size):
        stop = min(start + batch_size, n_events)
        batch = {key: value[start:stop] for key, value in chunk.items() if isinstance(value, np.ndarray) and len(value) == n_events}
        sources = build_sources(batch, device)

        with torch.no_grad():
            prediction = model.forward(sources)

        assignment_indices = extract_predictions([
            np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
            for assignment in prediction.assignments
        ])

        detections = [torch.sigmoid(detection).detach().cpu().numpy().astype(FLOAT) for detection in prediction.detections]
        classifications = {
            key: torch.softmax(value, 1).detach().cpu().numpy().astype(FLOAT)
            for key, value in prediction.classifications.items()
        }
        regressions = {key: value.detach().cpu().numpy().astype(FLOAT) for key, value in prediction.regressions.items()}

        dummy_index = torch.arange(stop - start, device=device)
        assignment_probabilities = []
        for assignment_probability, assignment, symmetries in zip(
            prediction.assignments,
            assignment_indices,
            model.event_info.product_symbolic_groups.values(),
        ):
            probability = assignment_probability.__getitem__((dummy_index, *assignment.T))
            probability = symmetries.order() * torch.exp(probability)
            assignment_probabilities.append(probability.detach().cpu().numpy().astype(FLOAT))

        batch_arrays = {
            "spanet_topE_b_index": assignment_indices[0][:, 0].astype(INT),
            "spanet_topMu_b_index": assignment_indices[1][:, 0].astype(INT),
            "spanet_topE_assignment_probability": assignment_probabilities[0],
            "spanet_topMu_assignment_probability": assignment_probabilities[1],
            "spanet_topE_detection_probability": detections[0],
            "spanet_topMu_detection_probability": detections[1],
        }

        reco_key = "EVENT/reco_quality"
        if reco_key in classifications:
            reco = classifications[reco_key]
            batch_arrays["spanet_reco_quality_prob_partial"] = reco[:, 0]
            batch_arrays["spanet_reco_quality_prob_full"] = reco[:, 1] if reco.shape[1] > 1 else np.zeros(len(reco), dtype=FLOAT)
            batch_arrays["spanet_reco_quality_pred"] = np.argmax(reco, axis=1).astype(INT)

        for key in REGRESSION_KEYS:
            if key in regressions:
                batch_arrays[f"spanet_{key.split('/')[-1]}"] = regressions[key]

        for key, value in derived_masses(batch, batch_arrays).items():
            batch_arrays[key] = value

        for key, value in batch_arrays.items():
            outputs.setdefault(key, []).append(np.asarray(value))

    return {key: np.concatenate(values, axis=0) for key, values in outputs.items()}


def original_arrays(events: ak.Array, mask: ak.Array, branches: dict[str, str | None], mode: str) -> dict[str, ak.Array]:
    if mode == "none":
        return {}

    if mode == "minimal":
        output: dict[str, np.ndarray] = {}
        n_events = int(ak.sum(mask))
        selected = {name: events[name][mask] for name in events.fields}
        for key in ("run", "lumi", "event", "weight"):
            branch = branches[key]
            if branch is None:
                continue
            dtype = INT if key in {"run", "lumi", "event"} else FLOAT
            output[branch] = as_numpy(scalar(selected, branch, n_events, 0), dtype=dtype)
        return output

    output = {}
    for name in events.fields:
        output[name] = ak.fill_none(events[name][mask], 0, axis=None)
    return output


def add_truth_helpers(output: dict[str, ak.Array | np.ndarray], chunk: dict[str, np.ndarray]) -> None:
    output["spanet_truth_reco_quality"] = chunk["reco_quality"].astype(INT)
    output["spanet_truth_topE_b_index"] = chunk["top_e_target"].astype(INT)
    output["spanet_truth_topMu_b_index"] = chunk["top_mu_target"].astype(INT)
    output["spanet_truth_topE_mask"] = chunk["top_e_mask"].astype(bool)
    output["spanet_truth_topMu_mask"] = chunk["top_mu_mask"].astype(bool)
    output["spanet_truth_available"] = chunk["truth_available"].astype(bool)
    output["spanet_split"] = chunk["split"].astype(np.int8)


def entry_range(total_entries: int, rank: int, world_size: int) -> tuple[int, int]:
    if world_size < 1:
        raise ValueError("--world-size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError("--rank must satisfy 0 <= rank < --world-size")
    start = total_entries * rank // world_size
    stop = total_entries * (rank + 1) // world_size
    return start, stop


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
    warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group` or `barrier `.*", category=UserWarning)

    files = expand_inputs(args.input)
    with uproot.open(files[0]) as root_file:
        branches = resolve_branches(root_file[args.tree], args)

    total_entries = 0
    for file_path in files:
        with uproot.open(file_path) as root_file:
            total_entries += root_file[args.tree].num_entries
    shard_start, shard_stop = entry_range(total_entries, args.rank, args.world_size)
    shard_entries = shard_stop - shard_start
    total_chunks = math.ceil(shard_entries / args.chunk_events)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = load_model(
        args.log_dir,
        testing_file=args.reference_h5,
        event_info_file=args.event_info,
        batch_size=args.batch_size,
        cuda=False,
        checkpoint=args.checkpoint,
    )
    model = model.to(device).eval()

    decompression_workers, interpretation_workers, compute_workers = worker_counts(args)
    print(
        f"Input entries: {total_entries}; shard={args.rank}/{args.world_size}; "
        f"entry_range=[{shard_start}, {shard_stop}); chunks={total_chunks}; "
        f"device={device}; batch_size={args.batch_size}; "
        f"original_branches={args.original_branches}"
    )
    print(
        "Parallelism: "
        f"decompression={decompression_workers}, "
        f"interpretation={interpretation_workers}, "
        f"compute={compute_workers}"
    )

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    read_branches = None if args.original_branches == "all" else sorted({branch for branch in branches.values() if branch is not None})
    infer_args = source_args(args)
    raw_offset = shard_start
    selected_total = 0
    tree_created = False

    limits = threadpool_limits(limits=compute_workers) if threadpool_limits is not None else nullcontext()
    with limits:
        with ThreadPoolExecutor(max_workers=decompression_workers) as decompression_executor:
            with ThreadPoolExecutor(max_workers=interpretation_workers) as interpretation_executor:
                iterator = uproot.iterate(
                    [f"{path}:{args.tree}" for path in files],
                    expressions=read_branches,
                    entry_start=shard_start,
                    entry_stop=shard_stop,
                    step_size=args.chunk_events,
                    library="ak",
                    decompression_executor=decompression_executor,
                    interpretation_executor=interpretation_executor,
                )
                with uproot.recreate(output_path) as output_file:
                    for events in tqdm(iterator, desc="ROOT inference", total=total_chunks, unit="chunk"):
                        mask = selection_mask(events, branches)
                        n_selected = int(ak.sum(mask))
                        first_branch = events.fields[0]
                        chunk_start = raw_offset
                        raw_offset += len(events[first_branch])
                        if n_selected == 0:
                            continue

                        chunk = make_chunk(events, branches, infer_args, chunk_start)
                        if chunk is None:
                            continue
                        if args.max_events is not None:
                            remaining = args.max_events - selected_total
                            if remaining <= 0:
                                break
                            if chunk["jets"].shape[0] > remaining:
                                chunk = {key: value[:remaining] for key, value in chunk.items()}
                                n_selected = remaining

                        predictions = predict_chunk(model, chunk, device, args.batch_size)
                        output_arrays: dict[str, ak.Array | np.ndarray] = original_arrays(
                            events, mask, branches, args.original_branches
                        )
                        if args.max_events is not None and n_selected != int(ak.sum(mask)):
                            output_arrays = {key: value[:n_selected] for key, value in output_arrays.items()}

                        output_arrays.update(predictions)
                        add_truth_helpers(output_arrays, chunk)

                        if not tree_created:
                            output_file[args.output_tree] = output_arrays
                            tree_created = True
                        else:
                            output_file[args.output_tree].extend(output_arrays)

                        selected_total += n_selected
                        if args.max_events is not None and selected_total >= args.max_events:
                            break

    print(f"Wrote {selected_total} selected events to {output_path}:{args.output_tree}")


if __name__ == "__main__":
    main()
