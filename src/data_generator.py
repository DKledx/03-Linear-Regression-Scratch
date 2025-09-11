"""
Data generator cho Linear Regression experiments
"""

from typing import Tuple, Optional
import numpy as np


class DataGenerator:
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def linear(self, num_samples: int = 200, weight: float = 2.0, bias: float = 3.0, noise_std: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        X = np.random.uniform(-10, 10, size=(num_samples, 1))
        noise = np.random.normal(0, noise_std, size=(num_samples,))
        y = weight * X.squeeze() + bias + noise
        return X, y

    def multiple(self, num_samples: int = 200, weights: Optional[np.ndarray] = None, bias: float = 5.0,
                 noise_std: float = 1.0, n_features: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        if weights is None:
            weights = np.array([3.0, 2.0, -1.0][:n_features])
            if len(weights) < n_features:
                weights = np.pad(weights, (0, n_features - len(weights)), 'constant')
        X = np.random.uniform(-10, 10, size=(num_samples, n_features))
        noise = np.random.normal(0, noise_std, size=(num_samples,))
        y = X @ weights + bias + noise
        return X, y

    def nonlinear_quadratic(self, num_samples: int = 200, noise_std: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        X = np.random.uniform(-10, 10, size=(num_samples, 1))
        noise = np.random.normal(0, noise_std, size=(num_samples,))
        y = (X.squeeze() ** 2) + 2 * X.squeeze() + 1 + noise
        return X, y


