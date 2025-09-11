"""
Gradient Descent utilities

Các biến thể của Gradient Descent để huấn luyện Linear Regression từ đầu:
- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-batch Gradient Descent
"""

from typing import List, Tuple, Optional
import numpy as np


class GradientDescent:
    """Batch Gradient Descent cho bài toán hồi quy"""

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6,
                 random_state: Optional[int] = None):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        if random_state is not None:
            np.random.seed(random_state)

    def optimize(self, X: np.ndarray, y: np.ndarray, initial_theta: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Tối ưu hàm MSE cho Linear Regression theo batch
        
        Returns:
            theta: vector trọng số tối ưu
            cost_history: lịch sử cost
        """
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape

        theta = initial_theta if initial_theta is not None else np.random.normal(0, 0.01, n)
        cost_history: List[float] = []
        prev_cost = float('inf')

        for _ in range(self.max_iterations):
            y_pred = X @ theta
            residual = y_pred - y
            cost = np.mean(residual ** 2)
            cost_history.append(cost)

            if abs(prev_cost - cost) < self.tolerance:
                break

            gradient = (1 / m) * (X.T @ residual)
            theta = theta - self.learning_rate * gradient
            prev_cost = cost

        return theta, cost_history


class StochasticGradientDescent:
    """Stochastic Gradient Descent (SGD) cho hồi quy"""

    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 100, shuffle: bool = True,
                 random_state: Optional[int] = None):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.shuffle = shuffle
        if random_state is not None:
            np.random.seed(random_state)

    def optimize(self, X: np.ndarray, y: np.ndarray, initial_theta: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float]]:
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape

        theta = initial_theta if initial_theta is not None else np.random.normal(0, 0.01, n)
        cost_history: List[float] = []

        for _ in range(self.max_epochs):
            indices = np.random.permutation(m) if self.shuffle else np.arange(m)
            total_cost = 0.0
            for i in indices:
                x_i = X[i:i + 1]
                y_i = y[i:i + 1]
                y_pred = x_i @ theta
                residual = y_pred - y_i
                total_cost += float((residual ** 2).mean())
                gradient = x_i.T @ residual
                theta = theta - self.learning_rate * gradient
            cost_history.append(total_cost / m)

        return theta, cost_history


class MiniBatchGradientDescent:
    """Mini-batch Gradient Descent cho hồi quy"""

    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 100, batch_size: int = 32,
                 shuffle: bool = True, random_state: Optional[int] = None):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        if random_state is not None:
            np.random.seed(random_state)

    def optimize(self, X: np.ndarray, y: np.ndarray, initial_theta: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float]]:
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape

        theta = initial_theta if initial_theta is not None else np.random.normal(0, 0.01, n)
        cost_history: List[float] = []

        for _ in range(self.max_epochs):
            indices = np.random.permutation(m) if self.shuffle else np.arange(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_cost = 0.0
            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_pred = X_batch @ theta
                residual = y_pred - y_batch
                epoch_cost += float(np.mean(residual ** 2))
                gradient = (1 / len(X_batch)) * (X_batch.T @ residual)
                theta = theta - self.learning_rate * gradient

            cost_history.append(epoch_cost / max(1, (m // self.batch_size)))

        return theta, cost_history


