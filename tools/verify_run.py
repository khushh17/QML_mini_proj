"""Check that saved main-run checkpoints reproduce metrics and respect shared data."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from pathlib import Path
import sys
import json
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from housing.experiment import evaluate, load_run


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not test_result.wasSuccessful():
        raise SystemExit('Correctness tests failed; checkpoint verification stopped.')
    run = ROOT / 'results' / 'main'
    previous = json.loads((run / 'results.json').read_text())
    with np.load(run / 'predictions.npz') as saved:
        old_predictions = {k: saved[k].copy() for k in saved.files}
    evaluate(run)
    current = json.loads((run / 'results.json').read_text())
    assert len(previous) == len(current) == 9, 'Expected 2 classical + 6 main neural + 1 ablation runs'
    for old, new in zip(previous, current):
        assert (old['model'], old['seed']) == (new['model'], new['seed'])
        for metric in ['rmse', 'mae', 'r2']:
            np.testing.assert_allclose(old[metric], new[metric], atol=1e-10, rtol=1e-10)
    with np.load(run / 'predictions.npz') as saved:
        for key, original in old_predictions.items():
            np.testing.assert_allclose(saved[key], original, atol=1e-10, rtol=1e-10)
    x, y, prep, splits, records = load_run(run)
    ids = np.concatenate(list(splits.values()))
    assert len(ids) == len(np.unique(ids))
    assert len(splits['test']) == 4128
    np.testing.assert_allclose(prep['x_mean'], x[splits['train']].mean(0))
    np.testing.assert_allclose(prep['y_mean'], y[splits['train']].mean())
    assert all(r['circuit_parameter_change_l2'] > 0 for r in records if r['model'].startswith('qnn'))
    verification = {
        'unit_tests_passed': test_result.testsRun,
        'checkpoint_predictions_reproduced': True,
        'all_nine_run_metrics_reproduced': True,
        'shared_split_disjointness': True,
        'preprocessing_matches_training_subset': True,
        'all_saved_quantum_circuits_updated': True,
        'test_rows': 4128,
        'comparison_tolerance': 1e-10,
    }
    (run / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
