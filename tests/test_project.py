"""Meaningful checks: units, leakage, gradients, quantum behavior, checkpoint parity."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
import housing  # Configure the writable plotting cache before PennyLane imports it.
import pennylane as qml
from housing.data import load_data, make_splits, fit_preprocessing, transform
from housing.models import QuantumRegressor, TabularResNet, predict_neural
from housing.experiment import metrics

torch.set_num_threads(2)


class DataTests(unittest.TestCase):
    def test_dataset_units_and_first_row(self):
        x, y, audit = load_data()
        self.assertEqual(x.shape, (20640, 8))
        np.testing.assert_allclose(x[0], [8.3252, 41, 880/126, 129/126, 322, 322/126, 37.88, -122.23])
        self.assertAlmostEqual(y[0], 4.526)
        self.assertEqual(audit['missing_or_nonfinite'], 0)

    def test_split_disjointness_and_reproducibility(self):
        a, b = make_splits(20640), make_splits(20640)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(len(set(np.concatenate(list(a.values())))), sum(map(len, a.values())))
        self.assertEqual(len(a['test']), 4128)

    def test_preprocessing_does_not_learn_from_held_out_rows(self):
        x, y, _ = load_data()
        s = make_splits(len(y))
        p = fit_preprocessing(x, y, s['train'])
        altered_x, altered_y = x.copy(), y.copy()
        altered_x[s['test']] = 1e10
        altered_y[s['val']] = -1e10
        q = fit_preprocessing(altered_x, altered_y, s['train'])
        for k in p:
            np.testing.assert_array_equal(p[k], q[k])
        np.testing.assert_allclose(transform(x, p)[s['train']].mean(0), 0, atol=1e-5)

    def test_metrics_and_dollar_conversion(self):
        result = metrics(np.array([1., 2., 3.]), np.array([2., 3., 4.]))
        self.assertAlmostEqual(result['rmse'], 1)
        self.assertAlmostEqual(result['mae_dollars'], 100000)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)

    def test_quantum_gradient_matches_finite_difference(self):
        model = QuantumRegressor(2).double()
        x = torch.randn(3, 8, dtype=torch.float64)
        model(x).sum().backward()
        gradient = model.theta.grad.clone()
        index = np.unravel_index(gradient.abs().argmax().item(), gradient.shape)
        self.assertGreater(abs(gradient[index].item()), 1e-5)
        eps = 1e-5
        with torch.no_grad():
            original = model.theta[index].item()
            model.theta[index] = original + eps
            plus = model(x).sum().item()
            model.theta[index] = original - eps
            minus = model(x).sum().item()
            model.theta[index] = original
        self.assertAlmostEqual(gradient[index].item(), (plus-minus)/(2*eps), places=5)

    def test_circuit_zero_state_and_batched_predictions(self):
        model = QuantumRegressor(2)
        with torch.no_grad():
            z = torch.stack(model.circuit(torch.zeros(8), torch.zeros_like(model.theta)))
        np.testing.assert_allclose(z.numpy(), np.ones(8), atol=1e-7)
        x = np.random.default_rng(5).normal(size=(7, 8)).astype('float32')
        np.testing.assert_allclose(predict_neural(model, x, 7), predict_neural(model, x, 1), atol=1e-6)

    def test_optimizer_updates_circuit_and_can_fit_tiny_batch(self):
        model = QuantumRegressor(1)
        x = torch.randn(8, 8)
        y = torch.linspace(-0.5, 0.5, 8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        initial = model.theta.detach().clone()
        first = torch.mean((model(x) - y)**2).item()
        for _ in range(25):
            optimizer.zero_grad()
            loss = torch.mean((model(x) - y)**2)
            loss.backward()
            optimizer.step()
        self.assertLess(torch.mean((model(x) - y)**2).item(), first)
        self.assertGreater(torch.linalg.vector_norm(model.theta.detach()-initial).item(), 0.01)

    def test_saved_models_roundtrip(self):
        x = np.ones((3, 8), dtype='float32')
        (Path(__file__).resolve().parents[1] / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as d:
            for factory in [TabularResNet, QuantumRegressor]:
                model = factory()
                original = predict_neural(model, x)
                path = Path(d) / 'weights.pt'
                torch.save(model.state_dict(), path)
                loaded = factory()
                loaded.load_state_dict(torch.load(path, weights_only=True))
                np.testing.assert_array_equal(original, predict_neural(loaded, x))


if __name__ == '__main__':
    unittest.main()
