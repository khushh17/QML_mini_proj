"""Draw the project's actual QNN and compare the saved test results. No training."""
import json
from pathlib import Path

import housing  # Set the project's writable Matplotlib cache before imports.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pennylane as qml
import torch

from housing.models import QuantumRegressor

RUN = Path(__file__).resolve().parent / "results" / "main"
OUT = RUN / "figures"
BLUE = "#315a6b"


def draw_circuit():
    model = QuantumRegressor(layers=2)
    # Draw the QNode itself. Angles are symbolic in this diagram, so the input
    # and initial weight values don't affect the gate sequence shown.
    fig, ax = qml.draw_mpl(model.circuit, decimals=None)(torch.zeros(8), model.theta)
    fig.set_size_inches(16, 6.5)
    ax.set_position([.035, .15, .94, .67])
    fig.text(.04, .94, "California Housing  /  Eight-qubit regression circuit",
             fontsize=19, weight="bold", color=BLUE)
    fig.text(.04, .88, "Read left to right. Each wire starts in |0⟩. The same eight features enter both layers.", fontsize=12)
    fig.text(.04, .08, "Each layer:  RY(input angle)  →  RY(learned angle)  →  RX(learned angle)  →  CNOT ring (0→1 through 7→0).", fontsize=12)
    fig.text(.04, .035, "Readout: measure ⟨Z⟩ on each wire. The classical output combines the eight measurements into one price prediction.", fontsize=12)
    fig.savefig(OUT / "qnn_gates.png", dpi=160, facecolor="white")
    plt.close(fig)


def draw_comparison():
    records = {r["model"]: r for r in json.loads((RUN / "summary.json").read_text())}
    config = json.loads((RUN / "config.json").read_text())
    models = [("mean", "Mean predictor"), ("boosted_trees", "Boosted trees"),
              ("resnet", "Tabular ResNet"), ("qnn", "QNN · two layers"),
              ("qnn_1layer", "QNN · one layer*")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.subplots_adjust(left=.25, right=.96, bottom=.23, top=.79)
    maximum = max(records[key]["rmse_mean"] for key, _ in models) * 100_000
    for i, (key, _) in enumerate(models):
        row = records[key]
        value = row["rmse_mean"] * 100_000
        ax.barh(i, value, height=.57, color=BLUE if key.startswith("qnn") else "#929799")
        spread = (row["rmse_std"] or 0) * 100_000
        if spread:
            ax.errorbar(value, i, xerr=spread, fmt="none", color="#222222", capsize=3)
        ax.text(value + spread + maximum*.012, i, f"${value:,.0f}", va="center")
    ax.set_yticks(range(len(models)), [label for _, label in models])
    ax.invert_yaxis()
    ax.set_xlim(0, maximum*1.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.set_xlabel("Test RMSE in historical dollars  ·  lower is better")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(.04, .92, "Classical and quantum models", fontsize=19, weight="bold", color=BLUE)
    fig.text(.04, .85, f"California Housing  /  Same {config['train_rows']:,} training rows and {config['test_rows']:,} held-out test rows", fontsize=11)
    fig.text(.04, .06, "Whiskers: one sample standard deviation across three seeds for ResNet and the two-layer QNN.", fontsize=9)
    fig.text(.04, .025, "*One-layer QNN: exploratory single run. Mean predictor and boosted trees also ran once.", fontsize=9)
    fig.savefig(OUT / "classical_vs_quantum.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    draw_circuit()
    draw_comparison()
    print(f"Saved both diagrams to {OUT}")
