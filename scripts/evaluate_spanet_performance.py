#!/usr/bin/env python3
"""Run SPANET inference on the held-out split and plot core performance metrics."""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.metrics import auc, confusion_matrix, roc_curve

from spanet.dataset.types import Source
from spanet.evaluation import evaluate_on_test_dataset, load_model


EVENT_PARTICLES = ("TopE", "TopMu")
CLASSIFICATION_KEY = "EVENT/reco_quality"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, help="Lightning/SPANET run version directory.")
    parser.add_argument("--input", required=True, help="Full HDF5 file containing metadata/split.")
    parser.add_argument("--event-info", default="configs/ttbar_dilep_event.yaml")
    parser.add_argument("--output-dir", required=True, help="Directory for plots, tables, and shard files.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to best epoch checkpoint.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--split-code", type=int, default=2, help="metadata/split code to evaluate. Default 2=test.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers per rank.")
    parser.add_argument("--gpus", type=int, default=1, help="Local GPUs to use. Spawns one worker per GPU.")
    parser.add_argument("--master-port", type=str, default="29571", help="Local distributed port for spawned workers.")
    return parser.parse_args()


def distributed_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def spawned_worker(local_rank: int, args: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", args.master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(args.gpus)
    run(args)


def recursive_copy_subset(src_group: h5py.Group, dst_group: h5py.Group, indices: np.ndarray, n_events: int) -> None:
    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            recursive_copy_subset(item, dst_group.create_group(name), indices, n_events)
            continue

        data = item[:]
        if data.shape and data.shape[0] == n_events:
            data = data[indices]
        dst_group.create_dataset(name, data=data, compression="lzf")


def create_subset_file(input_file: str, output_file: Path, split_code: int, rank: int, world_size: int, max_events: int | None) -> int:
    with h5py.File(input_file, "r") as src:
        split = src["metadata/split"][:]
        indices = np.flatnonzero(split == split_code)
        if max_events is not None:
            indices = indices[:max_events]
        indices = indices[rank::world_size]
        n_events = len(split)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_file, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["source_file"] = input_file
            dst.attrs["source_split_code"] = split_code
            dst.attrs["distributed_rank"] = rank
            dst.attrs["distributed_world_size"] = world_size
            recursive_copy_subset(src, dst, indices, n_events)

    return len(indices)


def assignment_truth(dataset) -> dict[str, np.ndarray]:
    truth = {}
    for name, (targets, masks) in dataset.assignments.items():
        truth[f"{name}_target"] = targets[:, 0].cpu().numpy()
        truth[f"{name}_mask"] = masks.cpu().numpy().astype(bool)
    return truth


def regression_truth(dataset) -> dict[str, np.ndarray]:
    return {f"{key}_true": value.cpu().numpy() for key, value in dataset.regressions.items()}


def classification_truth(dataset) -> dict[str, np.ndarray]:
    return {f"{key}_true": value.cpu().numpy() for key, value in dataset.classifications.items()}


def evaluate_rank(args: argparse.Namespace, rank: int, local_rank: int, world_size: int) -> Path:
    output_dir = Path(args.output_dir)
    shard_h5 = output_dir / "shards" / f"test_split_rank{rank:03d}.h5"
    n_events = create_subset_file(args.input, shard_h5, args.split_code, rank, world_size, args.max_events)
    print(f"[rank {rank}] evaluating {n_events} events from split {args.split_code}")

    model = load_model(
        args.log_dir,
        testing_file=str(shard_h5),
        event_info_file=args.event_info,
        batch_size=args.batch_size,
        cuda=False,
        checkpoint=args.checkpoint,
    )
    model.options.num_dataloader_workers = args.workers
    model.options.num_gpu = 1
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        model = model.cuda(local_rank)
    model.eval()

    with torch.no_grad():
        progress = None if world_size > 1 else __import__("rich").progress
        evaluation = evaluate_on_test_dataset(model, progress=progress)

    arrays = {}
    arrays.update(assignment_truth(model.testing_dataset))
    arrays.update(regression_truth(model.testing_dataset))
    arrays.update(classification_truth(model.testing_dataset))

    for name in EVENT_PARTICLES:
        arrays[f"{name}_pred"] = evaluation.assignments[name][:, 0]
        arrays[f"{name}_assignment_probability"] = evaluation.assignment_probabilities[name]
        arrays[f"{name}_detection_probability"] = evaluation.detection_probabilities[name]

    for key, values in evaluation.regressions.items():
        arrays[f"{key}_pred"] = values

    for key, values in evaluation.classifications.items():
        arrays[f"{key}_prob"] = values
        arrays[f"{key}_pred"] = np.argmax(values, axis=1)

    shard_npz = output_dir / "shards" / f"predictions_rank{rank:03d}.npz"
    np.savez_compressed(shard_npz, **arrays)
    return shard_npz


def concatenate_shards(output_dir: Path, world_size: int) -> dict[str, np.ndarray]:
    arrays_by_key: dict[str, list[np.ndarray]] = {}
    for rank in range(world_size):
        shard = np.load(output_dir / "shards" / f"predictions_rank{rank:03d}.npz")
        for key in shard.files:
            arrays_by_key.setdefault(key, []).append(shard[key])
    return {key: np.concatenate(values, axis=0) for key, values in arrays_by_key.items()}


def masked_accuracy(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(pred[mask] == target[mask]))


def plot_confusion(path: Path, target: np.ndarray, pred: np.ndarray, mask: np.ndarray, title: str) -> None:
    matrix = confusion_matrix(target[mask], pred[mask], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted jet")
    ax.set_ylabel("Truth jet")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_hist(path: Path, values: np.ndarray, title: str, xlabel: str, bins: int = 80) -> None:
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, histtype="step", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(path: Path, target: np.ndarray, score: np.ndarray) -> float:
    valid = target >= 0
    if len(np.unique(target[valid])) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(target[valid], score[valid])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Reco-quality ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return float(roc_auc)


def make_summary_and_plots(arrays: dict[str, np.ndarray], output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    summary = {"events": int(len(arrays["TopE_target"]))}

    full_mask = np.ones(summary["events"], dtype=bool)
    for name in EVENT_PARTICLES:
        target = arrays[f"{name}_target"]
        pred = arrays[f"{name}_pred"]
        mask = arrays[f"{name}_mask"]
        full_mask &= mask
        summary[f"{name}_accuracy"] = masked_accuracy(pred, target, mask)
        summary[f"{name}_valid_events"] = int(mask.sum())
        plot_confusion(plots / f"{name}_confusion.png", target, pred, mask, f"{name} assignment")
        plot_hist(plots / f"{name}_assignment_probability.png", arrays[f"{name}_assignment_probability"], f"{name} assignment probability", "probability")
        plot_hist(plots / f"{name}_detection_probability.png", arrays[f"{name}_detection_probability"], f"{name} detection probability", "probability")

    both_correct = (
        (arrays["TopE_pred"] == arrays["TopE_target"]) &
        (arrays["TopMu_pred"] == arrays["TopMu_target"]) &
        full_mask
    )
    summary["full_assignment_accuracy"] = float(np.mean(both_correct[full_mask])) if np.any(full_mask) else float("nan")
    summary["fully_matched_truth_events"] = int(full_mask.sum())

    reco_true_key = f"{CLASSIFICATION_KEY}_true"
    reco_prob_key = f"{CLASSIFICATION_KEY}_prob"
    reco_pred_key = f"{CLASSIFICATION_KEY}_pred"
    if reco_true_key in arrays:
        reco_true = arrays[reco_true_key]
        reco_pred = arrays[reco_pred_key]
        reco_score = arrays[reco_prob_key][:, 1]
        valid = reco_true >= 0
        summary["reco_quality_accuracy"] = float(np.mean(reco_pred[valid] == reco_true[valid]))
        summary["reco_quality_auc"] = plot_roc(plots / "reco_quality_roc.png", reco_true, reco_score)
        plot_confusion(plots / "reco_quality_confusion.png", reco_true, reco_pred, valid, "Reco-quality classification")
        plot_hist(plots / "reco_quality_score.png", reco_score, "Reco-quality score", "P(full)")

    regression_summary = {}
    for key in sorted(k[:-5] for k in arrays if k.endswith("_pred") and k.startswith("EVENT/")):
        if key == CLASSIFICATION_KEY:
            continue
        pred = arrays[f"{key}_pred"]
        true = arrays.get(f"{key}_true")
        if true is None:
            continue
        valid = np.isfinite(true)
        residual = pred[valid] - true[valid]
        regression_summary[key] = {
            "valid": int(valid.sum()),
            "bias": float(np.mean(residual)),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual * residual))),
        }
        safe_name = key.replace("/", "_")
        plot_hist(plots / f"{safe_name}_residual.png", residual, f"{key} residual", "prediction - truth")
    summary["regressions"] = regression_summary

    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run(args: argparse.Namespace) -> None:
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
    warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group` or `barrier `.*", category=UserWarning)
    rank, local_rank, world_size = distributed_info()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "shards").mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()

    evaluate_rank(args, rank, local_rank, world_size)

    if world_size > 1:
        torch.distributed.barrier()

    if rank == 0:
        for _ in range(600):
            if all((output_dir / "shards" / f"predictions_rank{r:03d}.npz").exists() for r in range(world_size)):
                break
            time.sleep(1)
        arrays = concatenate_shards(output_dir, world_size)
        make_summary_and_plots(arrays, output_dir)

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main() -> None:
    args = parse_args()
    if "WORLD_SIZE" not in os.environ and args.gpus > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpus > 1 requested, but CUDA is not available.")
        available = torch.cuda.device_count()
        if args.gpus > available:
            raise RuntimeError(f"Requested {args.gpus} GPUs, but only {available} are visible.")
        mp.spawn(spawned_worker, args=(args,), nprocs=args.gpus, join=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
