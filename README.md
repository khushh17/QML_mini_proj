# California Housing: Classical and Quantum Regression

A CPU-based comparison of classical regression and an eight-qubit hybrid quantum neural network (QNN). Inspired by *Revisiting Deep Learning Models for Tabular Data*, this project adds a quantum approach; it does not reproduce the full paper benchmark or demonstrate quantum advantage.

## Project structure

```text
CaliforniaHousing/     Supplied dataset and column descriptions
housing/               Data processing, models, training, evaluation and plots
references/            Assigned research paper
results/main/          Recorded metrics, splits, predictions, models and figures
output/pdf/            Project report for submission
tests/                 Correctness tests
tools/                 Report builder and saved-run verification
run.py                 Main command-line entry point
generate_diagrams.py   Circuit image and model-comparison chart
terminal_circuit.py    Formatted circuit diagram in the terminal
demo.cmd               Windows shortcut for saved-model prediction
requirements.txt       Pinned project dependencies
requirements-lock.txt  Recorded Windows environment
```

The report is `output/pdf/Final_Quantum_Computing_Project_Report.pdf`. Project repository: [khushh17/QML_mini_proj](https://github.com/khushh17/QML_mini_proj). Old study guides, duplicate images and smoke-run outputs have been removed. Main-run evidence is retained so the report can be checked and the models reused.

## Setup

Use Python 3.11 or 3.12; the recorded experiment used Python 3.12.14. On Windows, from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

On macOS, create the environment with `python3.12 -m venv .venv`, then run `.venv/bin/python -m pip install -r requirements.txt`. Use `.venv/bin/python` instead of the Windows interpreter path below. Recreate the environment on each computer; do not copy or upload `.venv`.

No GPU or quantum hardware is required. PennyLane simulates eight qubits on the CPU, with two compute threads used during training. The dataset is already supplied locally; no download is needed.

## Run

```powershell
# Predict one held-out example using saved models (no training)
.\.venv\Scripts\python.exe run.py demo

# Trace a quantum prediction through encoding, measurement and readout
.\.venv\Scripts\python.exe run.py explain

# Draw the circuit in the terminal or save the two diagram images
.\.venv\Scripts\python.exe terminal_circuit.py --ascii --width 58
.\.venv\Scripts\python.exe generate_diagrams.py

# Rebuild the submission report from saved results and circuit image
.\.venv\Scripts\python.exe tools/build_final_report.py

# Optional: check correctness or train a new run
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run.py train --output results/repeat
```

For `demo` and `explain`, use `--index 25` to choose another test example. New training outputs must use a new or empty folder; the saved main experiment is protected. New runs are ignored by Git unless explicitly added.

## Experiment and results

The California Housing dataset contains 20,640 census block-group records. Eight derived features describe income, age, rooms, bedrooms, population, occupancy and location. The target is historical median housing value, not a current individual house price.

Every model uses the same 2,048 training, 512 validation and 4,128 test examples; 13,952 rows are unused to keep compute modest. Preprocessing is fitted on training data only. Neural training allows 40 epochs, with validation-based early stopping. ResNet and the main QNN each use seeds 7, 17 and 27.

| Model | Test RMSE (historical dollars) | R2 | Mean training seconds |
| --- | ---: | ---: | ---: |
| Mean predictor | 114,569 | -0.0017 | 0.001 |
| Boosted trees | 52,278 | 0.7914 | 0.4 |
| Tabular ResNet | 58,011 | 0.7432 | 5.9 |
| QNN, two layers | 88,468 | 0.4027 | 116.9 |
| QNN, one layer (exploratory) | 108,619 | 0.0997 | 74.2 |

Lower RMSE is better. Neural main-model rows average three runs; the other rows use one run. The QNN improves RMSE by 22.8% over the mean predictor, but the trained classical models perform better. Times are local measurements, not guarantees on other machines.

The QNN uses angle encoding `2 * atan(z)`, trainable RY/RX rotations, a CNOT ring and eight Pauli-Z expectations followed by a linear readout. The main circuit has two layers, 64 gates, logical depth 22 and 41 total trainable parameters. It uses PennyLane `default.qubit`, exact expectations (`shots=None`) and PyTorch backpropagation. No noise or real-device experiment was performed.

The recorded verification in `results/main/verification.json` reports eight passing tests and reproduced predictions/metrics for all nine fits. `tools/verify_run.py` can rerun these checks; it also refreshes evaluation files.

## Repository contents

`.gitignore` excludes `.venv`, `.env` secrets, caches, temporary files and extra experiments. It deliberately keeps the small trained models, primary results, supplied data, reference paper and submission report. Ignoring files prevents new Git additions; it does not delete local installations or remove files already tracked by Git.

## References

- [Assigned paper](https://arxiv.org/abs/2106.11959)
- [Paper authors' code](https://github.com/yandex-research/tabular-dl-revisiting-models) (not this project's repository)
- [Dataset archive](https://figshare.com/articles/dataset/cal_housing_tgz/3829992)
- [Dataset conversion](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_california_housing.py)
- [PennyLane PyTorch interface](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html)
