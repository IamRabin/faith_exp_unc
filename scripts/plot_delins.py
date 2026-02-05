from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch


def load_curves(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")

    return {
        "deletion_curve": obj["deletion_curve"].float(),
        "insertion_curve": obj["insertion_curve"].float(),
        "deletion_curve_random": obj.get("deletion_curve_random", None),
        "insertion_curve_random": obj.get("insertion_curve_random", None),
        "auc_deletion": obj.get("auc_deletion", None),
        "auc_insertion": obj.get("auc_insertion", None),
        "auc_deletion_random": obj.get("auc_deletion_random", None),
        "auc_insertion_random": obj.get("auc_insertion_random", None),
    }


def fraction_axis(n: int):
    return torch.linspace(0.0, 1.0, steps=n).numpy()


def plot_single(
    series: List[Tuple[str, Dict[str, Any]]],
    mode: str,
    output_path: str,
    title: str,
    ylim: tuple[float, float] | None = (0.0, 1.0),
):
    """
    mode: 'insertion' or 'deletion'
    """
    assert mode in ["insertion", "deletion"]

    plt.figure(figsize=(4.5, 3.6))
    ax = plt.gca()

    for label, data in series:
        curve = data[f"{mode}_curve"].numpy()
        curve_rand = data.get(f"{mode}_curve_random", None)

        x = fraction_axis(len(curve))

        ax.plot(
            x,
            curve,
            label=label,
            linewidth=2.5,
        )

        if curve_rand is not None:
            ax.plot(
                x,
                curve_rand.numpy(),
                linestyle="--",
                linewidth=2.0,
                alpha=0.8,
                label=f"{label} (rand)",
            )

    ax.set_xlabel("Fraction of features modified")
    ax.set_ylabel("Confidence on true class")
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)

    # ---- Legend placement ----
    if mode == "insertion":
        ax.legend(loc="upper left", frameon=True)
    else:  # deletion
        ax.legend(loc="upper right", frameon=True)

    ax.grid(False)
    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--ylim", type=str, default="0,1")
    args = parser.parse_args()

    if len(args.labels) != len(args.inputs):
        raise ValueError(f"--labels and --inputs must have same length, got {len(args.labels)} vs {len(args.inputs)}")

    os.makedirs(args.output_dir, exist_ok=True)

    series = []
    for lbl, path in zip(args.labels, args.inputs):
        series.append((lbl, load_curves(path)))

    lo, hi = map(float, args.ylim.split(","))

    plot_single(
        series,
        mode="insertion",
        output_path=os.path.join(args.output_dir, "delins_insertion.pdf"),
        title=args.title,
        ylim=(lo, hi),
    )

    plot_single(
        series,
        mode="deletion",
        output_path=os.path.join(args.output_dir, "delins_deletion.pdf"),
        title=args.title,
        ylim=(lo, hi),
    )


if __name__ == "__main__":
    main()
