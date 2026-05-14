"""Plot training curves from CSV metrics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path) -> dict[str, list[float]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        columns: dict[str, list[float]] = {col: [] for col in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                try:
                    columns[key].append(float(value))
                except (ValueError, TypeError):
                    columns[key].append(float("nan"))
    return columns


def plot(run_dir: Path, save: bool = False) -> None:
    training_path = run_dir / "training_metrics.csv"
    eval_path = run_dir / "eval_metrics.csv"

    if not training_path.exists():
        raise FileNotFoundError(f"No training metrics at {training_path}")

    train = load_csv(training_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Dino DQN Training", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(train["episode"], train["avg_score_25"], color="#2563eb", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score (25-ep avg)")
    ax.set_title("Score Over Training")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(train["episode"], train["epsilon"], color="#dc2626", linewidth=1.2)
    if "advisor" in train:
        ax.plot(train["episode"], train["advisor"], color="#f59e0b", linewidth=1.2, label="advisor")
        ax.legend(["epsilon", "advisor"], fontsize=8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_title("Exploration")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    losses = [v for v in train.get("loss", []) if v == v]
    loss_eps = [e for e, v in zip(train["episode"], train.get("loss", [])) if v == v]
    if losses:
        ax.plot(loss_eps, losses, color="#7c3aed", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if eval_path.exists():
        ev = load_csv(eval_path)
        ax.plot(ev["episode"], ev["avg_score"], color="#059669", marker="o", markersize=4, linewidth=1.5)
        ax.plot(ev["episode"], ev["best_score"], color="#059669", linestyle="--", alpha=0.5, linewidth=1)
        ax.legend(["avg", "best"], fontsize=8)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        ax.set_title("Evaluation Scores")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No eval data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Evaluation Scores")

    plt.tight_layout()
    if save:
        out = run_dir / "training_curves.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves.")
    parser.add_argument("run_dir", type=Path, help="Path to the run directory containing CSV files.")
    parser.add_argument("--save", action="store_true", help="Save plot as PNG instead of showing it.")
    args = parser.parse_args()
    plot(args.run_dir, save=args.save)


if __name__ == "__main__":
    main()
