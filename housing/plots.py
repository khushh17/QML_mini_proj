"""Generate presentation figures from saved results, never invented numbers."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'mean': 'Mean baseline', 'boosted_trees': 'Boosted trees',
          'resnet': 'Tabular ResNet', 'qnn': 'QNN (2 layers)', 'qnn_1layer': 'QNN (1 layer)'}


def make_plots(output):
    output = Path(output)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 140})
    summary = json.loads((output / 'summary.json').read_text())
    main = [r for r in summary if r['model'] != 'qnn_1layer']
    fig, ax = plt.subplots(figsize=(7.4, 3.6), layout='constrained')
    ax.bar([LABELS[r['model']] for r in main], [r['rmse_mean'] * 100000 for r in main],
           yerr=[(r['rmse_std'] or 0) * 100000 for r in main], capsize=4,
           color=['#a0a0a0', '#666666', '#999999', '#315a6b'])
    ax.set_ylabel('Test RMSE (historical dollars)')
    ax.set_title('Same training rows and held-out test set; lower is better')
    fig.savefig(figures / 'comparison.png')
    plt.close(fig)
    pred = np.load(output / 'predictions.npz')
    records = json.loads((output / 'training_records.json').read_text())
    seed = next(r['seed'] for r in records if r['model'] == 'qnn')
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), layout='constrained')
    for ax, name in zip(axes.flat, ['mean', 'boosted_trees', 'resnet', 'qnn']):
        chosen = seed if name in ('resnet', 'qnn') else 7
        ax.scatter(pred['actual'], pred[f'{name}_{chosen}'], s=3, alpha=0.16, color='#315a6b', rasterized=True)
        ax.plot([0, 5.5], [0, 5.5], color='#555555', linestyle='--', linewidth=1)
        ax.set(title=LABELS[name], xlabel='Actual ($100,000)', ylabel='Predicted ($100,000)')
    fig.savefig(figures / 'predictions.png')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), layout='constrained')
    for ax, name in zip(axes, ['resnet', 'qnn']):
        history = json.loads((output / f'history_{name}_{seed}.json').read_text())
        ax.plot([h['epoch'] for h in history], [h['train_mse_standardized'] for h in history], label='Training', color='#777777')
        ax.plot([h['epoch'] for h in history], [h['val_mse_standardized'] for h in history], label='Validation', color='#315a6b')
        ax.set(title=LABELS[name], xlabel='Epoch', ylabel='MSE (standardized target)')
        ax.legend(frameon=False)
    fig.savefig(figures / 'learning_curves.png')
    plt.close(fig)
    # Full circuit, not a decorative approximation.
    import torch
    import pennylane as qml
    from .models import QuantumRegressor
    model = QuantumRegressor(2)
    fig, _ = qml.draw_mpl(model.circuit, decimals=None)(torch.zeros(8), model.theta)
    fig.set_size_inches(12, 5)
    fig.savefig(figures / 'quantum_circuit.png', bbox_inches='tight', dpi=160)
    plt.close(fig)
    # Dataset target and distribution, clearly distinguished from model outcomes.
    from .data import load_data
    _, y, _ = load_data()
    fig, ax = plt.subplots(figsize=(7.4, 3.0), layout='constrained')
    ax.hist(y * 100000, bins=45, color='#777777', edgecolor='white', linewidth=0.3)
    ax.set(xlabel='Historical median house value ($)', ylabel='Block groups',
           title='Target distribution: concentration at the recorded upper limit')
    fig.savefig(figures / 'target_distribution.png')
    plt.close(fig)
    print(f'Saved figures to {figures}')
