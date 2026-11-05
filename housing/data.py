"""Convert the supplied StatLib rows to scikit-learn's eight-feature layout."""
from pathlib import Path
import hashlib
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'CaliforniaHousing' / 'cal_housing.data'
FEATURES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude']


def load_data(path=DATA):
    raw = np.loadtxt(path, delimiter=',', dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 9 or not np.isfinite(raw).all():
        raise ValueError('Expected a finite numeric matrix with nine raw columns.')
    if np.any(raw[:, 6] <= 0):
        raise ValueError('Household counts must be positive before division.')
    # Raw: longitude, latitude, age, rooms, bedrooms, population,
    # households, median income, median value in dollars.
    x = np.column_stack([raw[:, 7], raw[:, 2], raw[:, 3] / raw[:, 6],
                         raw[:, 4] / raw[:, 6], raw[:, 5],
                         raw[:, 5] / raw[:, 6], raw[:, 1], raw[:, 0]])
    y = raw[:, 8] / 100_000.0
    audit = {
        'rows': len(y), 'features': FEATURES, 'raw_columns': 9,
        'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'missing_or_nonfinite': int((~np.isfinite(raw)).sum()),
        'duplicate_raw_rows': int(len(raw) - len(np.unique(raw, axis=0))),
        'target_min_dollars': float(raw[:, 8].min()),
        'target_max_dollars': float(raw[:, 8].max()),
        'rows_at_maximum': int(np.sum(raw[:, 8] == raw[:, 8].max())),
        'first_features': x[0].tolist(), 'first_target': float(y[0]),
    }
    return x, y, audit


def make_splits(n, train_size=2048, val_size=512):
    # Fixed split seed, separate from model initialization seeds.
    pool, test = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    train_pool, val_pool = train_test_split(pool, test_size=0.2, random_state=42)
    rng = np.random.default_rng(42)
    train_pool = rng.permutation(train_pool)
    val_pool = rng.permutation(val_pool)
    if train_size > len(train_pool) or val_size > len(val_pool):
        raise ValueError('Requested subset exceeds its original split.')
    train = train_pool if train_size == 0 else train_pool[:train_size]
    val = val_pool if val_size == 0 else val_pool[:val_size]
    return {'train': train, 'val': val, 'test': test}


def fit_preprocessing(x, y, train):
    # Only these selected training rows determine scaling, including target scaling.
    scale = x[train].std(axis=0)
    scale[scale == 0] = 1
    return {'x_mean': x[train].mean(axis=0), 'x_scale': scale,
            'y_mean': np.array(y[train].mean()),
            'y_scale': np.array(max(y[train].std(), 1e-12))}


def transform(x, prep):
    return ((x - prep['x_mean']) / prep['x_scale']).astype(np.float32)
