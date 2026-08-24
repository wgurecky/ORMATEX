#!/usr/bin/env python3
"""Plot Navier-Stokes example fields stored as x,y,u,v,p CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FILES = (
    "navier_stokes_lid_driven_cavity.csv",
    "navier_stokes_lid_driven_cavity_comp.csv",
    "navier_stokes_pipe.csv",
    "navier_stokes_cylinder.csv",
)


def read_fields(path: Path) -> np.ndarray:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"x", "y", "u", "v", "p"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path} must have columns x,y,u,v,p")
        rows = [[float(row[name]) for name in ("x", "y", "u", "v", "p")] for row in reader]
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return np.asarray(rows)


def final_time(path: Path) -> float | None:
    probe = path.with_name(f"{path.stem}_probe.csv")
    if not probe.is_file():
        return None
    with probe.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return float(rows[-1]["t"]) if rows else None


def contour(ax, data: np.ndarray, column: int, title: str) -> None:
    values = data[:, column]
    valid = np.isfinite(data[:, [0, 1, column]]).all(axis=1)
    if valid.sum() < 3:
        ax.text(0.5, 0.5, "insufficient finite data", ha="center", va="center")
        ax.set_title(title)
        return
    plot = ax.tricontourf(data[valid, 0], data[valid, 1], values[valid], levels=30, cmap="viridis")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.figure.colorbar(plot, ax=ax)


def vector_plot(ax, data: np.ndarray, max_vectors: int) -> None:
    valid = np.isfinite(data[:, :4]).all(axis=1)
    points = data[valid]
    if len(points) > max_vectors:
        indices = np.linspace(0, len(points) - 1, max_vectors, dtype=int)
        points = points[indices]
    ax.quiver(points[:, 0], points[:, 1], points[:, 2], points[:, 3])
    ax.set_title("Velocity vectors")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_file(path: Path, output_dir: Path, max_vectors: int) -> Path:
    data = read_fields(path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    contour(axes[0, 0], data, 2, "u velocity")
    contour(axes[0, 1], data, 3, "v velocity")
    contour(axes[1, 0], data, 4, "Pressure")
    vector_plot(axes[1, 1], data, max_vectors)
    title = path.stem
    if path.name == "navier_stokes_cylinder.csv":
        time = final_time(path)
        title = "Cylinder final-time fields"
        if time is not None:
            title += f" (t={time:g})"
    fig.suptitle(title)
    output = output_dir / f"{path.stem}.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="spatial CSV files; defaults to Navier-Stokes files in target/",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "target",
        help="directory containing default CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for PNG files (default: target/navier_stokes_plots)",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=500,
        help="maximum velocity arrows per plot",
    )
    args = parser.parse_args()
    if args.max_vectors < 1:
        parser.error("--max-vectors must be positive")

    files = args.files or [
        args.target / name for name in DEFAULT_FILES if (args.target / name).is_file()
    ]
    if not files:
        parser.error("no Navier-Stokes spatial CSV files found in " + str(args.target))
    missing = [path for path in files if not path.is_file()]
    if missing:
        parser.error("missing CSV file(s): " + ", ".join(map(str, missing)))
    output_dir = args.output_dir or args.target / "navier_stokes_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        print(plot_file(path, output_dir, args.max_vectors))


if __name__ == "__main__":
    main()
