"""Train with validation only, then evaluate saved checkpoints on a held-out test set."""
from pathlib import Path
import csv
import importlib.metadata
import json
import platform
import time
import joblib
import numpy as np
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
from .data import load_data, make_splits, fit_preprocessing, transform
from .models import build_model, predict_neural


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def metrics(y, pred):
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite model predictions.')
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    return {'rmse': rmse, 'mae': mae, 'r2': float(r2_score(y, pred)),
            'rmse_dollars': rmse * 100000, 'mae_dollars': mae * 100000}


def train_neural(name, seed, x_train, y_train, x_val, y_val, epochs, batch_size):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(name)
    lr = 0.02 if name.startswith('qnn') else 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt, yt = torch.from_numpy(x_train), torch.tensor(y_train, dtype=torch.float32)
    best = float('inf')
    best_state = None
    best_epoch, stale = 0, 0
    history = []
    initial = model.theta.detach().clone() if name.startswith('qnn') else None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = rng.permutation(len(xt))
        total = 0.0
        for offset in range(0, len(xt), batch_size):
            idx = permutation[offset:offset + batch_size]
            # BatchNorm requires at least two rows while training.
            if len(idx) == 1 and name == 'resnet':
                model.eval()
            optimizer.zero_grad()
            loss = torch.mean((model(xt[idx]) - yt[idx]) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError('Training loss became nonfinite.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.train()
            total += loss.item() * len(idx)
        val_mse = float(np.mean((predict_neural(model, x_val) - y_val) ** 2))
        history.append({'epoch': epoch, 'train_mse_standardized': total / len(xt),
                        'val_mse_standardized': val_mse})
        if val_mse < best - 1e-6:
            best, best_epoch, stale = val_mse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f'{name} seed={seed} epoch={epoch}: validation MSE={val_mse:.4f}', flush=True)
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    info = {'train_seconds': time.perf_counter() - start, 'best_epoch': best_epoch,
            'epochs_run': epoch, 'learning_rate': lr,
            'parameters': sum(p.numel() for p in model.parameters()),
            'validation_mse_standardized': best}
    if initial is not None:
        info['circuit_parameter_change_l2'] = float(torch.linalg.vector_norm(model.theta.detach() - initial))
        info['circuit_parameters'] = model.theta.numel()
        info['qubits'] = 8
        info['layers'] = model.layers
    return model, history, info


def run_training(output, train_size=2048, val_size=512, epochs=40,
                 seeds=(7, 17, 27), ablation=True, threads=2):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty. Choose a new --output directory to preserve results.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'models').mkdir(exist_ok=True)
    torch.set_num_threads(threads)
    x, y, audit = load_data()
    splits = make_splits(len(y), train_size, val_size)
    np.savez(output / 'splits.npz', **splits)
    prep = fit_preprocessing(x, y, splits['train'])
    np.savez(output / 'preprocessing.npz', **prep)
    write_json(output / 'data_audit.json', audit)
    write_json(output / 'config.json', {
        'train_rows': len(splits['train']), 'validation_rows': len(splits['val']),
        'test_rows': len(splits['test']), 'unused_rows': len(y) - sum(map(len, splits.values())),
        'model_seeds': list(seeds), 'split_seed': 42, 'max_epochs': epochs,
        'batch_size': 64, 'patience': 10, 'threads': threads, 'ablation': ablation,
        'main_qnn_layers': 2, 'tuning': 'fixed configurations; no hyperparameter search',
        'python': platform.python_version(), 'platform': platform.platform(),
        'processor': platform.processor(),
        'packages': {n: importlib.metadata.version(n) for n in
                     ['numpy', 'scipy', 'scikit-learn', 'torch', 'pennylane', 'matplotlib']}})
    z = transform(x, prep)
    ys = (y - prep['y_mean']) / prep['y_scale']
    tr, va = splits['train'], splits['val']
    records = []
    with threadpool_limits(limits=threads):
        # Deterministic fixed tree and mean settings: repeat fits would be identical.
        for name, estimator in [
            ('mean', DummyRegressor(strategy='mean')),
            ('boosted_trees', HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15,
                 learning_rate=0.08, l2_regularization=1.0, early_stopping=False, random_state=7))]:
            start = time.perf_counter()
            estimator.fit(x[tr], y[tr])
            seconds = time.perf_counter() - start
            joblib.dump(estimator, output / 'models' / f'{name}_7.joblib')
            records.append({'model': name, 'seed': 7, 'train_seconds': seconds,
                            'validation_rmse': metrics(y[va], estimator.predict(x[va]))['rmse']})
            print(f'{name}: trained in {seconds:.2f}s', flush=True)
        for name in ['resnet', 'qnn'] + (['qnn_1layer'] if ablation else []):
            for seed in (seeds[:1] if name == 'qnn_1layer' else seeds):
                model, history, info = train_neural(name, seed, z[tr], ys[tr], z[va], ys[va], epochs, 64)
                torch.save(model.state_dict(), output / 'models' / f'{name}_{seed}.pt')
                write_json(output / f'history_{name}_{seed}.json', history)
                records.append({'model': name, 'seed': seed, **info})
                write_json(output / 'training_records.json', records)
    write_json(output / 'training_records.json', records)
    # First test-set model evaluation happens only after all training is complete.
    return evaluate(output, threads)


def load_run(output):
    output = Path(output)
    x, y, audit = load_data()
    expected = json.loads((output / 'data_audit.json').read_text())['sha256']
    if audit['sha256'] != expected:
        raise ValueError('Dataset changed since training; refusing mismatched evaluation.')
    with np.load(output / 'preprocessing.npz') as saved:
        prep = dict(saved)
    with np.load(output / 'splits.npz') as saved:
        splits = dict(saved)
    records = json.loads((output / 'training_records.json').read_text())
    return x, y, prep, splits, records


def predict_saved(output, name, seed, x, prep):
    path = Path(output) / 'models' / f'{name}_{seed}'
    if name in ('mean', 'boosted_trees'):
        # Only load model files produced by this project; joblib is not safe for untrusted files.
        return joblib.load(path.with_suffix('.joblib')).predict(x)
    model = build_model(name)
    model.load_state_dict(torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True))
    return predict_neural(model, transform(x, prep)) * prep['y_scale'] + prep['y_mean']


def evaluate(output, threads=2):
    output = Path(output)
    torch.set_num_threads(threads)
    x, y, prep, splits, records = load_run(output)
    test = splits['test']
    predictions = {'row_id': test, 'actual': y[test]}
    results = []
    with threadpool_limits(limits=threads):
        for record in records:
            name, seed = record['model'], record['seed']
            start = time.perf_counter()
            pred = predict_saved(output, name, seed, x[test], prep)
            predictions[f'{name}_{seed}'] = pred
            results.append({**record, **metrics(y[test], pred),
                            'test_predict_seconds_including_load': time.perf_counter() - start})
    np.savez(output / 'predictions.npz', **predictions)
    write_json(output / 'results.json', results)
    columns = ['model', 'seed', 'rmse', 'mae', 'r2', 'rmse_dollars', 'mae_dollars', 'train_seconds']
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    summary = []
    for name in dict.fromkeys(r['model'] for r in results):
        group = [r for r in results if r['model'] == name]
        row = {'model': name, 'runs': len(group)}
        for metric in ['rmse', 'mae', 'r2', 'train_seconds']:
            values = [r[metric] for r in group]
            row[metric + '_mean'] = float(np.mean(values))
            row[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    write_json(output / 'summary.json', summary)
    return summary
