#!/usr/bin/env python3
"""Convert dileptonic ttbar prod_v2 ROOT ntuples into SPANET-ready HDF5."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import awkward as ak
import h5py
import numpy as np
import uproot
import vector
from tqdm import tqdm

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

vector.register_awkward()


FLOAT = np.float32
INT = np.int64


@dataclass(frozen=True)
class BranchSpec:
    name: str
    aliases: tuple[str, ...]


DEFAULT_BRANCHES = {
    "jet_pt": ("Jet_pt", "SelectedJet_pt", "jet_pt"),
    "jet_eta": ("Jet_eta", "SelectedJet_eta", "jet_eta"),
    "jet_phi": ("Jet_phi", "SelectedJet_phi", "jet_phi"),
    "jet_mass": ("Jet_mass", "SelectedJet_mass", "jet_mass"),
    "electron_pt": ("Electron_pt", "El_pt", "el_pt", "electron_pt"),
    "electron_eta": ("Electron_eta", "El_eta", "el_eta", "electron_eta"),
    "electron_phi": ("Electron_phi", "El_phi", "el_phi", "electron_phi"),
    "electron_mass": ("Electron_mass", "El_mass", "el_mass", "electron_mass"),
    "electron_charge": ("Electron_charge", "El_charge", "el_charge", "electron_charge"),
    "muon_pt": ("Muon_pt", "Mu_pt", "mu_pt", "muon_pt"),
    "muon_eta": ("Muon_eta", "Mu_eta", "mu_eta", "muon_eta"),
    "muon_phi": ("Muon_phi", "Mu_phi", "mu_phi", "muon_phi"),
    "muon_mass": ("Muon_mass", "Mu_mass", "mu_mass", "muon_mass"),
    "muon_charge": ("Muon_charge", "Mu_charge", "mu_charge", "muon_charge"),
    "met_pt": ("MET_pt", "PuppiMET_pt", "met_pt"),
    "met_phi": ("MET_phi", "PuppiMET_phi", "met_phi"),
    "weight": ("eventWeight", "genWeight", "weight"),
    "run": ("run", "Run", "runNumber"),
    "lumi": ("luminosityBlock", "lumi", "LumiBlock"),
    "event": ("event", "Event", "eventNumber"),
    "truth_available": ("gen_ttbar_truth_available", "truth_available"),
    "gen_b_pt": ("GenB_pt", "gen_b_pt", "b_pt"),
    "gen_b_eta": ("GenB_eta", "gen_b_eta", "b_eta"),
    "gen_b_phi": ("GenB_phi", "gen_b_phi", "b_phi"),
    "gen_b_mass": ("GenB_mass", "gen_b_mass", "b_mass"),
    "gen_bbar_pt": ("GenBbar_pt", "GenBBar_pt", "gen_bbar_pt", "bbar_pt"),
    "gen_bbar_eta": ("GenBbar_eta", "GenBBar_eta", "gen_bbar_eta", "bbar_eta"),
    "gen_bbar_phi": ("GenBbar_phi", "GenBBar_phi", "gen_bbar_phi", "bbar_phi"),
    "gen_bbar_mass": ("GenBbar_mass", "GenBBar_mass", "gen_bbar_mass", "bbar_mass"),
    "nu_px": ("GenNu_px", "gen_nu_px", "nu_px"),
    "nu_py": ("GenNu_py", "gen_nu_py", "nu_py"),
    "nu_pz": ("GenNu_pz", "gen_nu_pz", "nu_pz"),
    "nu_pt": ("GenNu_pt", "gen_nu_pt", "nu_pt"),
    "nu_eta": ("GenNu_eta", "gen_nu_eta", "nu_eta"),
    "nu_phi": ("GenNu_phi", "gen_nu_phi", "nu_phi"),
    "nu_mass": ("GenNu_mass", "gen_nu_mass", "nu_mass"),
    "nubar_px": ("GenNuBar_px", "GenNubar_px", "gen_nubar_px", "nubar_px"),
    "nubar_py": ("GenNuBar_py", "GenNubar_py", "gen_nubar_py", "nubar_py"),
    "nubar_pz": ("GenNuBar_pz", "GenNubar_pz", "gen_nubar_pz", "nubar_pz"),
    "nubar_pt": ("GenNuBar_pt", "GenNubar_pt", "gen_nubar_pt", "nubar_pt"),
    "nubar_eta": ("GenNuBar_eta", "GenNubar_eta", "gen_nubar_eta", "nubar_eta"),
    "nubar_phi": ("GenNuBar_phi", "GenNubar_phi", "gen_nubar_phi", "nubar_phi"),
    "nubar_mass": ("GenNuBar_mass", "GenNubar_mass", "gen_nubar_mass", "nubar_mass"),
    "top_px": ("GenTop_px", "gen_top_px", "top_px"),
    "top_py": ("GenTop_py", "gen_top_py", "top_py"),
    "top_pz": ("GenTop_pz", "gen_top_pz", "top_pz"),
    "top_e": ("GenTop_e", "GenTop_E", "gen_top_e", "top_e"),
    "top_pt": ("GenTop_pt", "gen_top_pt", "top_pt"),
    "top_eta": ("GenTop_eta", "gen_top_eta", "top_eta"),
    "top_phi": ("GenTop_phi", "gen_top_phi", "top_phi"),
    "top_mass": ("GenTop_mass", "gen_top_mass", "top_mass"),
    "tbar_px": ("GenTbar_px", "GenAntiTop_px", "gen_tbar_px", "tbar_px"),
    "tbar_py": ("GenTbar_py", "GenAntiTop_py", "gen_tbar_py", "tbar_py"),
    "tbar_pz": ("GenTbar_pz", "GenAntiTop_pz", "gen_tbar_pz", "tbar_pz"),
    "tbar_e": ("GenTbar_e", "GenTbar_E", "GenAntiTop_E", "gen_tbar_e", "tbar_e"),
    "tbar_pt": ("GenTbar_pt", "GenAntiTop_pt", "gen_tbar_pt", "tbar_pt"),
    "tbar_eta": ("GenTbar_eta", "GenAntiTop_eta", "gen_tbar_eta", "tbar_eta"),
    "tbar_phi": ("GenTbar_phi", "GenAntiTop_phi", "gen_tbar_phi", "tbar_phi"),
    "tbar_mass": ("GenTbar_mass", "GenAntiTop_mass", "gen_tbar_mass", "tbar_mass"),
}


def comma_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def visible_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Input ROOT files or glob patterns.")
    parser.add_argument("--output", required=True, help="Output HDF5 path.")
    parser.add_argument("--tree", default="Events", help="TTree name.")
    parser.add_argument(
        "--chunk-events",
        type=int,
        default=200_000,
        help="Approximate number of input events to read per ROOT chunk.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Total CPU workers. Defaults to all CPUs visible to the process.",
    )
    parser.add_argument(
        "--decompression-workers",
        type=int,
        default=None,
        help="Threads for ROOT basket decompression. Defaults to half of --workers.",
    )
    parser.add_argument(
        "--interpretation-workers",
        type=int,
        default=None,
        help="Threads for ROOT array interpretation. Defaults to remaining --workers.",
    )
    parser.add_argument(
        "--compute-workers",
        type=int,
        default=None,
        help="Thread limit for NumPy/BLAS-backed compute. Defaults to --workers.",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Stop after this many selected events.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--pair-dr-max", type=float, default=0.4, help="Maximum dR for gen b/bbar to jet matching.")
    parser.add_argument("--compression", default="lzf", choices=("gzip", "lzf", "none"), help=argparse.SUPPRESS)

    for key, aliases in DEFAULT_BRANCHES.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}-branches",
            default=",".join(aliases),
            type=comma_list,
            help=f"Comma-separated aliases for {key}.",
        )

    return parser.parse_args()


def expand_inputs(patterns: Iterable[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(matches if matches else [pattern])
    missing = [path for path in files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Input file(s) do not exist: {missing[:5]}")
    return files


def resolve_branches(tree: uproot.TTree, args: argparse.Namespace) -> dict[str, str | None]:
    available = set(tree.keys())
    resolved: dict[str, str | None] = {}
    for key in DEFAULT_BRANCHES:
        aliases = getattr(args, f"{key}_branches")
        resolved[key] = next((branch for branch in aliases if branch in available), None)
    required = [
        "jet_pt",
        "jet_eta",
        "jet_phi",
        "jet_mass",
        "electron_pt",
        "electron_eta",
        "electron_phi",
        "electron_mass",
        "electron_charge",
        "muon_pt",
        "muon_eta",
        "muon_phi",
        "muon_mass",
        "muon_charge",
        "met_pt",
        "met_phi",
    ]
    missing = [key for key in required if resolved[key] is None]
    if missing:
        raise KeyError(f"Missing required branches: {missing}. Available examples: {sorted(available)[:30]}")
    return resolved


def selected_first(array: ak.Array) -> ak.Array:
    return ak.firsts(array)


def as_numpy(array: ak.Array, dtype=FLOAT, fill=0) -> np.ndarray:
    return ak.to_numpy(ak.fill_none(array, fill)).astype(dtype, copy=False)


def scalar(events: dict[str, ak.Array], branch: str | None, n_events: int, fill: float = 0) -> ak.Array:
    if branch is None:
        return ak.Array(np.full(n_events, fill))
    values = events[branch]
    if values.ndim > 1:
        values = selected_first(values)
    return ak.fill_none(values, fill)


def delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi


def delta_r(eta1: np.ndarray, phi1: np.ndarray, eta2: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return np.hypot(eta1 - eta2, delta_phi(phi1, phi2))


def invariant_mass(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2) -> np.ndarray:
    v1 = vector.array({"pt": pt1, "eta": eta1, "phi": phi1, "mass": mass1})
    v2 = vector.array({"pt": pt2, "eta": eta2, "phi": phi2, "mass": mass2})
    return np.asarray((v1 + v2).mass, dtype=FLOAT)


def components_from_pt_eta_phi_mass(pt, eta, phi, mass) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(np.maximum(px * px + py * py + pz * pz + mass * mass, 0.0))
    return px.astype(FLOAT), py.astype(FLOAT), pz.astype(FLOAT), energy.astype(FLOAT)


def target_pxpypz(
    selected: dict[str, ak.Array],
    branches: dict[str, str | None],
    prefix: str,
    n_events: int,
) -> tuple[np.ndarray, bool]:
    cartesian = (f"{prefix}_px", f"{prefix}_py", f"{prefix}_pz")
    if all(branches[key] is not None for key in cartesian):
        return np.stack([as_numpy(scalar(selected, branches[key], n_events)) for key in cartesian], axis=-1), True

    polar = (f"{prefix}_pt", f"{prefix}_eta", f"{prefix}_phi")
    if all(branches[key] is not None for key in polar):
        pt = as_numpy(scalar(selected, branches[f"{prefix}_pt"], n_events))
        eta = as_numpy(scalar(selected, branches[f"{prefix}_eta"], n_events))
        phi = as_numpy(scalar(selected, branches[f"{prefix}_phi"], n_events))
        mass_branch = branches.get(f"{prefix}_mass")
        mass = as_numpy(scalar(selected, mass_branch, n_events, 0.0)) if mass_branch is not None else np.zeros(n_events, dtype=FLOAT)
        px, py, pz, _ = components_from_pt_eta_phi_mass(pt, eta, phi, mass)
        return np.stack([px, py, pz], axis=-1), True

    return np.zeros((n_events, 3), dtype=FLOAT), False


def target_pxpypze(
    selected: dict[str, ak.Array],
    branches: dict[str, str | None],
    prefix: str,
    n_events: int,
) -> tuple[np.ndarray, bool]:
    cartesian = (f"{prefix}_px", f"{prefix}_py", f"{prefix}_pz", f"{prefix}_e")
    if all(branches[key] is not None for key in cartesian):
        return np.stack([as_numpy(scalar(selected, branches[key], n_events)) for key in cartesian], axis=-1), True

    polar = (f"{prefix}_pt", f"{prefix}_eta", f"{prefix}_phi", f"{prefix}_mass")
    if all(branches[key] is not None for key in polar):
        pt = as_numpy(scalar(selected, branches[f"{prefix}_pt"], n_events))
        eta = as_numpy(scalar(selected, branches[f"{prefix}_eta"], n_events))
        phi = as_numpy(scalar(selected, branches[f"{prefix}_phi"], n_events))
        mass = as_numpy(scalar(selected, branches[f"{prefix}_mass"], n_events))
        px, py, pz, energy = components_from_pt_eta_phi_mass(pt, eta, phi, mass)
        return np.stack([px, py, pz, energy], axis=-1), True

    return np.zeros((n_events, 4), dtype=FLOAT), False


def stable_split(run: np.ndarray, lumi: np.ndarray, event: np.ndarray, val_fraction: float, test_fraction: float) -> np.ndarray:
    split = np.zeros(len(event), dtype=np.int8)
    train_threshold = 1.0 - val_fraction - test_fraction
    val_threshold = 1.0 - test_fraction
    for i, key in enumerate(zip(run, lumi, event)):
        digest = hashlib.blake2b(f"{key[0]}:{key[1]}:{key[2]}".encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little") / 2**64
        if value >= val_threshold:
            split[i] = 2
        elif value >= train_threshold:
            split[i] = 1
    return split


def write_dataset(group: h5py.Group, name: str, data: np.ndarray, compression: str | None) -> None:
    group.create_dataset(name, data=data, compression=compression, shuffle=compression == "gzip")


def write_spanet_native(h5: h5py.File, data: dict[str, np.ndarray], compression: str | None) -> None:
    inputs = h5.create_group("INPUTS")
    jets = inputs.create_group("Jets")
    leptons = inputs.create_group("Leptons")
    met = inputs.create_group("Met")
    event = inputs.create_group("Event")

    n_events = data["jets"].shape[0]
    write_dataset(jets, "MASK", np.ones((n_events, 2), dtype=bool), compression)
    write_dataset(jets, "pt", data["jets"][:, :, 0], compression)
    write_dataset(jets, "eta", data["jets"][:, :, 1], compression)
    write_dataset(jets, "sin_phi", np.sin(data["jets"][:, :, 2]).astype(FLOAT), compression)
    write_dataset(jets, "cos_phi", np.cos(data["jets"][:, :, 2]).astype(FLOAT), compression)
    write_dataset(jets, "mass", data["jets"][:, :, 3], compression)

    write_dataset(leptons, "e_pt", data["leptons"][:, 0, 0], compression)
    write_dataset(leptons, "e_eta", data["leptons"][:, 0, 1], compression)
    write_dataset(leptons, "e_sin_phi", np.sin(data["leptons"][:, 0, 2]).astype(FLOAT), compression)
    write_dataset(leptons, "e_cos_phi", np.cos(data["leptons"][:, 0, 2]).astype(FLOAT), compression)
    write_dataset(leptons, "e_mass", data["leptons"][:, 0, 3], compression)
    write_dataset(leptons, "e_charge", data["leptons"][:, 0, 4], compression)
    write_dataset(leptons, "mu_pt", data["leptons"][:, 1, 0], compression)
    write_dataset(leptons, "mu_eta", data["leptons"][:, 1, 1], compression)
    write_dataset(leptons, "mu_sin_phi", np.sin(data["leptons"][:, 1, 2]).astype(FLOAT), compression)
    write_dataset(leptons, "mu_cos_phi", np.cos(data["leptons"][:, 1, 2]).astype(FLOAT), compression)
    write_dataset(leptons, "mu_mass", data["leptons"][:, 1, 3], compression)
    write_dataset(leptons, "mu_charge", data["leptons"][:, 1, 4], compression)

    write_dataset(met, "met", data["met"][:, 0], compression)
    write_dataset(met, "sin_phi", np.sin(data["met"][:, 1]).astype(FLOAT), compression)
    write_dataset(met, "cos_phi", np.cos(data["met"][:, 1]).astype(FLOAT), compression)

    event_feature_names = ["reserved", "dphi_emu", "dr_emu", "dphi_jj", "dr_jj", "m_jj", "m_ej_min", "m_muj_min"]
    for index, name in enumerate(event_feature_names):
        write_dataset(event, name, data["event_features"][:, index], compression)

    targets = h5.create_group("TARGETS")
    top_e = targets.create_group("TopE")
    top_mu = targets.create_group("TopMu")
    write_dataset(top_e, "b", data["top_e_target"], compression)
    write_dataset(top_e, "MASK", data["top_e_mask"], compression)
    write_dataset(top_mu, "b", data["top_mu_target"], compression)
    write_dataset(top_mu, "MASK", data["top_mu_mask"], compression)

    regressions = h5.create_group("REGRESSIONS")
    regression_event = regressions.create_group("EVENT")
    regression_values = {
        "nu_px": data["nu"][:, 0, 0],
        "nu_py": data["nu"][:, 0, 1],
        "nu_pz": data["nu"][:, 0, 2],
        "nubar_px": data["nu"][:, 1, 0],
        "nubar_py": data["nu"][:, 1, 1],
        "nubar_pz": data["nu"][:, 1, 2],
        "top_px": data["top"][:, 0, 0],
        "top_py": data["top"][:, 0, 1],
        "top_pz": data["top"][:, 0, 2],
        "top_e": data["top"][:, 0, 3],
        "tbar_px": data["top"][:, 1, 0],
        "tbar_py": data["top"][:, 1, 1],
        "tbar_pz": data["top"][:, 1, 2],
        "tbar_e": data["top"][:, 1, 3],
    }
    valid_regression = data["truth_available"].astype(bool)
    for name, values in regression_values.items():
        output = values.astype(FLOAT).copy()
        output[~valid_regression] = np.nan
        write_dataset(regression_event, name, output, compression)

    classifications = h5.create_group("CLASSIFICATIONS")
    classification_event = classifications.create_group("EVENT")
    write_dataset(classification_event, "reco_quality", data["reco_quality"].astype(INT), compression)


def make_chunk(events: dict[str, ak.Array], branches: dict[str, str | None], args: argparse.Namespace, start_index: int):
    n_raw = len(events[branches["jet_pt"]])
    n_jet = ak.num(events[branches["jet_pt"]])
    n_ele = ak.num(events[branches["electron_pt"]])
    n_mu = ak.num(events[branches["muon_pt"]])

    ele_charge = selected_first(events[branches["electron_charge"]])
    mu_charge = selected_first(events[branches["muon_charge"]])
    os_emu = (n_ele == 1) & (n_mu == 1) & (ele_charge * mu_charge < 0)
    selection = (n_jet == 2) & os_emu
    if not ak.any(selection):
        return None

    selected = {name: events[name][selection] for name in events.fields}
    n = int(ak.sum(selection))

    jet_pt = as_numpy(selected[branches["jet_pt"]])
    jet_eta = as_numpy(selected[branches["jet_eta"]])
    jet_phi = as_numpy(selected[branches["jet_phi"]])
    jet_mass = as_numpy(selected[branches["jet_mass"]])
    jets = np.stack([jet_pt, jet_eta, jet_phi, jet_mass], axis=-1)

    e_pt = as_numpy(selected_first(selected[branches["electron_pt"]]))
    e_eta = as_numpy(selected_first(selected[branches["electron_eta"]]))
    e_phi = as_numpy(selected_first(selected[branches["electron_phi"]]))
    e_mass = as_numpy(selected_first(selected[branches["electron_mass"]]))
    e_charge = as_numpy(selected_first(selected[branches["electron_charge"]]), dtype=FLOAT)
    mu_pt = as_numpy(selected_first(selected[branches["muon_pt"]]))
    mu_eta = as_numpy(selected_first(selected[branches["muon_eta"]]))
    mu_phi = as_numpy(selected_first(selected[branches["muon_phi"]]))
    mu_mass = as_numpy(selected_first(selected[branches["muon_mass"]]))
    mu_charge = as_numpy(selected_first(selected[branches["muon_charge"]]), dtype=FLOAT)
    leptons = np.stack(
        [
            np.stack([e_pt, e_eta, e_phi, e_mass, e_charge], axis=-1),
            np.stack([mu_pt, mu_eta, mu_phi, mu_mass, mu_charge], axis=-1),
        ],
        axis=1,
    )

    met = np.stack(
        [
            as_numpy(scalar(selected, branches["met_pt"], n)),
            as_numpy(scalar(selected, branches["met_phi"], n)),
        ],
        axis=-1,
    )

    dr_emu = delta_r(e_eta, e_phi, mu_eta, mu_phi)
    dphi_emu = delta_phi(e_phi, mu_phi)
    dr_jj = delta_r(jet_eta[:, 0], jet_phi[:, 0], jet_eta[:, 1], jet_phi[:, 1])
    dphi_jj = delta_phi(jet_phi[:, 0], jet_phi[:, 1])
    m_jj = invariant_mass(jet_pt[:, 0], jet_eta[:, 0], jet_phi[:, 0], jet_mass[:, 0], jet_pt[:, 1], jet_eta[:, 1], jet_phi[:, 1], jet_mass[:, 1])
    m_ej = np.stack(
        [
            invariant_mass(e_pt, e_eta, e_phi, e_mass, jet_pt[:, 0], jet_eta[:, 0], jet_phi[:, 0], jet_mass[:, 0]),
            invariant_mass(e_pt, e_eta, e_phi, e_mass, jet_pt[:, 1], jet_eta[:, 1], jet_phi[:, 1], jet_mass[:, 1]),
        ],
        axis=1,
    )
    m_muj = np.stack(
        [
            invariant_mass(mu_pt, mu_eta, mu_phi, mu_mass, jet_pt[:, 0], jet_eta[:, 0], jet_phi[:, 0], jet_mass[:, 0]),
            invariant_mass(mu_pt, mu_eta, mu_phi, mu_mass, jet_pt[:, 1], jet_eta[:, 1], jet_phi[:, 1], jet_mass[:, 1]),
        ],
        axis=1,
    )
    event_features = np.stack(
        [np.zeros(n), dphi_emu, dr_emu, dphi_jj, dr_jj, m_jj, np.min(m_ej, axis=1), np.min(m_muj, axis=1)],
        axis=-1,
    ).astype(FLOAT)
    truth_available = as_numpy(scalar(selected, branches["truth_available"], n, 1), dtype=bool, fill=False)

    pair_label = np.full(n, -1, dtype=INT)
    pair_mask = np.zeros(n, dtype=bool)
    top_e_target = np.full(n, -1, dtype=INT)
    top_mu_target = np.full(n, -1, dtype=INT)
    top_e_mask = np.zeros(n, dtype=bool)
    top_mu_mask = np.zeros(n, dtype=bool)
    if all(branches[key] is not None for key in ("gen_b_eta", "gen_b_phi", "gen_bbar_eta", "gen_bbar_phi")):
        b_eta = as_numpy(scalar(selected, branches["gen_b_eta"], n, np.nan))
        b_phi = as_numpy(scalar(selected, branches["gen_b_phi"], n, np.nan))
        bbar_eta = as_numpy(scalar(selected, branches["gen_bbar_eta"], n, np.nan))
        bbar_phi = as_numpy(scalar(selected, branches["gen_bbar_phi"], n, np.nan))
        dr_b = np.stack([delta_r(jet_eta[:, 0], jet_phi[:, 0], b_eta, b_phi), delta_r(jet_eta[:, 1], jet_phi[:, 1], b_eta, b_phi)], axis=1)
        dr_bbar = np.stack([delta_r(jet_eta[:, 0], jet_phi[:, 0], bbar_eta, bbar_phi), delta_r(jet_eta[:, 1], jet_phi[:, 1], bbar_eta, bbar_phi)], axis=1)
        b_jet = np.argmin(dr_b, axis=1)
        bbar_jet = np.argmin(dr_bbar, axis=1)
        b_match = np.min(dr_b, axis=1) < args.pair_dr_max
        bbar_match = np.min(dr_bbar, axis=1) < args.pair_dr_max
        valid_match = (b_jet != bbar_jet) & b_match & bbar_match & truth_available
        e_side_jet = np.where(e_charge > 0, b_jet, bbar_jet)
        e_side_match = np.where(e_charge > 0, b_match, bbar_match)
        e_side_dr = np.where(e_charge > 0, np.min(dr_b, axis=1), np.min(dr_bbar, axis=1))
        mu_side_jet = np.where(e_charge > 0, bbar_jet, b_jet)
        mu_side_match = np.where(e_charge > 0, bbar_match, b_match)
        mu_side_dr = np.where(e_charge > 0, np.min(dr_bbar, axis=1), np.min(dr_b, axis=1))

        top_e_target[e_side_match & truth_available] = e_side_jet[e_side_match & truth_available]
        top_mu_target[mu_side_match & truth_available] = mu_side_jet[mu_side_match & truth_available]
        top_e_mask = (top_e_target >= 0) & truth_available
        top_mu_mask = (top_mu_target >= 0) & truth_available

        collision = top_e_mask & top_mu_mask & (top_e_target == top_mu_target)
        keep_e_collision = collision & (e_side_dr <= mu_side_dr)
        keep_mu_collision = collision & (mu_side_dr < e_side_dr)
        top_e_mask[collision & ~keep_e_collision] = False
        top_mu_mask[collision & ~keep_mu_collision] = False
        top_e_target[~top_e_mask] = -1
        top_mu_target[~top_mu_mask] = -1

        pair_label[valid_match & (e_side_jet == 0)] = 0
        pair_label[valid_match & (e_side_jet == 1)] = 1
        pair_mask = top_e_mask & top_mu_mask & (top_e_target != top_mu_target)

    nu = np.zeros((n, 2, 3), dtype=FLOAT)
    nu_mask = np.zeros((n, 2), dtype=bool)
    for idx, prefix in enumerate(("nu", "nubar")):
        values, valid = target_pxpypz(selected, branches, prefix, n)
        nu[:, idx, :] = values
        nu_mask[:, idx] = valid & truth_available

    top = np.zeros((n, 2, 4), dtype=FLOAT)
    top_mask = np.zeros((n, 2), dtype=bool)
    for idx, prefix in enumerate(("top", "tbar")):
        values, valid = target_pxpypze(selected, branches, prefix, n)
        top[:, idx, :] = values
        top_mask[:, idx] = valid & truth_available

    full_truth = top_e_mask & top_mu_mask & np.all(nu_mask, axis=1) & np.all(top_mask, axis=1)
    reco_quality = full_truth.astype(INT)
    reco_quality[~truth_available] = -1
    reco_mask = truth_available.astype(bool)

    run = as_numpy(scalar(selected, branches["run"], n, 0), dtype=INT)
    lumi = as_numpy(scalar(selected, branches["lumi"], n, 0), dtype=INT)
    default_event = np.arange(start_index, start_index + n_raw, dtype=INT)[ak.to_numpy(selection)]
    if branches["event"] is None:
        event = default_event
    else:
        event = as_numpy(scalar(selected, branches["event"], n, 0), dtype=INT)

    split = stable_split(run, lumi, event, args.val_fraction, args.test_fraction)
    weight = as_numpy(scalar(selected, branches["weight"], n, 1.0))

    return {
        "jets": jets.astype(FLOAT),
        "leptons": leptons.astype(FLOAT),
        "met": met.astype(FLOAT),
        "event_features": event_features,
        "pair_label": pair_label,
        "top_e_target": top_e_target,
        "top_mu_target": top_mu_target,
        "reco_quality": reco_quality,
        "nu": nu,
        "top": top,
        "pair_mask": pair_mask,
        "top_e_mask": top_e_mask,
        "top_mu_mask": top_mu_mask,
        "nu_mask": nu_mask,
        "top_mask": top_mask,
        "reco_mask": reco_mask,
        "weight": weight,
        "run": run,
        "lumi": lumi,
        "event": event,
        "split": split,
        "truth_available": truth_available,
    }


def concatenate(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}


def worker_counts(args: argparse.Namespace) -> tuple[int, int, int]:
    workers = max(1, args.workers or visible_cpu_count())
    decompression_workers = args.decompression_workers
    interpretation_workers = args.interpretation_workers
    if decompression_workers is None and interpretation_workers is None:
        decompression_workers = max(1, workers // 2)
        interpretation_workers = max(1, workers - decompression_workers)
    elif decompression_workers is None:
        interpretation_workers = max(1, interpretation_workers)
        decompression_workers = max(1, workers - interpretation_workers)
    elif interpretation_workers is None:
        decompression_workers = max(1, decompression_workers)
        interpretation_workers = max(1, workers - decompression_workers)

    compute_workers = max(1, args.compute_workers or workers)
    return max(1, decompression_workers), max(1, interpretation_workers), compute_workers


def main() -> None:
    args = parse_args()
    files = expand_inputs(args.input)
    total_entries = 0
    with uproot.open(files[0]) as root_file:
        branches = resolve_branches(root_file[args.tree], args)
    for file_path in files:
        with uproot.open(file_path) as root_file:
            total_entries += root_file[args.tree].num_entries
    total_chunks = math.ceil(total_entries / args.chunk_events) if args.chunk_events > 0 else None

    decompression_workers, interpretation_workers, compute_workers = worker_counts(args)
    print(
        "Parallelism: "
        f"decompression={decompression_workers}, "
        f"interpretation={interpretation_workers}, "
        f"compute={compute_workers}; "
        f"chunk_events={args.chunk_events}; "
        f"compression={args.compression}"
    )
    print(f"Input entries: {total_entries}; expected chunks: {total_chunks}")

    read_branches = sorted({branch for branch in branches.values() if branch is not None})
    chunks = []
    raw_offset = 0
    limits = threadpool_limits(limits=compute_workers) if threadpool_limits is not None else nullcontext()
    with limits:
        with ThreadPoolExecutor(max_workers=decompression_workers) as decompression_executor:
            with ThreadPoolExecutor(max_workers=interpretation_workers) as interpretation_executor:
                iterator = uproot.iterate(
                    [f"{path}:{args.tree}" for path in files],
                    expressions=read_branches,
                    step_size=args.chunk_events,
                    library="ak",
                    decompression_executor=decompression_executor,
                    interpretation_executor=interpretation_executor,
                )
                for events in tqdm(iterator, desc="Converting ROOT chunks", total=total_chunks, unit="chunk"):
                    chunk = make_chunk(events, branches, args, raw_offset)
                    raw_offset += len(events[read_branches[0]])
                    if chunk is None:
                        continue
                    chunks.append(chunk)
                    if args.max_events is not None and sum(len(item["split"]) for item in chunks) >= args.max_events:
                        break

    if not chunks:
        raise RuntimeError("No events passed the OS e-mu exactly-two-jet selection.")

    data = concatenate(chunks)
    if args.max_events is not None:
        data = {key: value[: args.max_events] for key, value in data.items()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    with h5py.File(output_path, "w") as h5:
        write_spanet_native(h5, data, compression)

        inputs = h5.create_group("inputs")
        targets = h5.create_group("targets")
        masks = h5.create_group("masks")
        weights = h5.create_group("weights")
        metadata = h5.create_group("metadata")

        write_dataset(inputs, "jets", data["jets"], compression)
        write_dataset(inputs, "leptons", data["leptons"], compression)
        write_dataset(inputs, "met", data["met"], compression)
        write_dataset(inputs, "event", data["event_features"], compression)
        write_dataset(targets, "pair_label", data["pair_label"], compression)
        write_dataset(targets, "reco_quality", data["reco_quality"], compression)
        write_dataset(targets, "nu", data["nu"], compression)
        write_dataset(targets, "top", data["top"], compression)
        write_dataset(masks, "pair", data["pair_mask"], compression)
        write_dataset(masks, "reco", data["reco_mask"], compression)
        write_dataset(masks, "nu", data["nu_mask"], compression)
        write_dataset(masks, "top", data["top_mask"], compression)
        write_dataset(weights, "event", data["weight"], compression)
        write_dataset(metadata, "run", data["run"], compression)
        write_dataset(metadata, "luminosityBlock", data["lumi"], compression)
        write_dataset(metadata, "event", data["event"], compression)
        write_dataset(metadata, "split", data["split"], compression)
        write_dataset(metadata, "truth_available", data["truth_available"], compression)

        h5.attrs["schema_version"] = "ttbar-dilep-spanet-v1"
        h5.attrs["tree"] = args.tree
        h5.attrs["source_files"] = json.dumps(files)
        h5.attrs["resolved_branches"] = json.dumps(branches, sort_keys=True)
        h5.attrs["jet_features"] = json.dumps(["pt", "eta", "phi", "mass"])
        h5.attrs["lepton_features"] = json.dumps(["pt", "eta", "phi", "mass", "charge"])
        h5.attrs["met_features"] = json.dumps(["pt", "phi"])
        h5.attrs["event_features"] = json.dumps(["reserved", "dphi_emu", "dr_emu", "dphi_jj", "dr_jj", "m_jj", "m_ej_min", "m_muj_min"])
        h5.attrs["split_codes"] = json.dumps({"train": 0, "val": 1, "test": 2})
        h5.attrs["val_fraction"] = args.val_fraction
        h5.attrs["test_fraction"] = args.test_fraction

    counts = {name: int(np.sum(data["split"] == code)) for name, code in {"train": 0, "val": 1, "test": 2}.items()}
    print(f"Wrote {len(data['split'])} selected events to {output_path}")
    print(f"Splits: {counts}")
    print(f"Pair labels valid: {int(np.sum(data['pair_mask']))}")
    print(f"Full reco labels: {int(np.sum(data['reco_quality'] == 1))}")
    print(f"Partial reco labels: {int(np.sum(data['reco_quality'] == 0))}")


if __name__ == "__main__":
    main()
