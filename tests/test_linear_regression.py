import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.linear_regression import SimpleLinearRegression, MultipleLinearRegression


def test_simple_linear_regression_learns_parameters():
    # y = 2x + 3
    rng = np.random.default_rng(42)
    X = rng.uniform(-5, 5, size=(200, 1))
    noise = rng.normal(0, 0.2, size=(200,))
    y = 2 * X.squeeze() + 3 + noise

    model = SimpleLinearRegression(learning_rate=0.05, max_iterations=5000, tolerance=1e-10, random_state=42)
    model.fit(X, y)
    params = model.get_parameters()

    assert np.isclose(params['bias'], 3, atol=0.2)
    assert np.isclose(params['weights'][0], 2, atol=0.2)


def test_multiple_linear_regression_ridge_converges():
    # y = 3x1 + 2x2 - x3 + 5
    rng = np.random.default_rng(123)
    X = rng.uniform(-5, 5, size=(500, 3))
    noise = rng.normal(0, 0.3, size=(500,))
    true_w = np.array([3.0, 2.0, -1.0])
    y = X @ true_w + 5 + noise

    model = MultipleLinearRegression(learning_rate=0.05, max_iterations=4000, regularization='ridge', alpha=0.01,
                                     tolerance=1e-10, random_state=123)
    model.fit(X, y)
    params = model.get_parameters()

    assert np.isclose(params['bias'], 5, atol=0.3)
    assert np.allclose(params['weights'], true_w, atol=0.3)


