"""
Visualization utilities cho Linear Regression
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def plot_learning_curve(self, cost_history: List[float], title: str = "Learning Curve") -> None:
        plt.figure(figsize=(8, 5))
        plt.plot(cost_history)
        plt.title(title)
        plt.xlabel('Iterations')
        plt.ylabel('Cost (MSE)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_regression_line(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float,
                              title: str = "Regression Line") -> None:
        if X.ndim > 1 and X.shape[1] != 1:
            raise ValueError("plot_regression_line chỉ áp dụng cho dữ liệu 1 chiều")
        X = X.reshape(-1, 1)
        y_pred = X @ weights.reshape(-1, 1) + bias
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, label='Data', alpha=0.7)
        plt.plot(X, y_pred, color='red', label='Prediction')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


