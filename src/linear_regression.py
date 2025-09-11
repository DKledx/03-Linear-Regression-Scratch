"""
Linear Regression Implementation from Scratch

Implement các thuật toán Linear Regression từ đầu để hiểu sâu về:
- Cost functions
- Gradient descent
- Vectorization
- Optimization
"""

import numpy as np
from typing import Tuple, List, Optional
import warnings


class SimpleLinearRegression:
    """
    Simple Linear Regression implementation từ đầu
    
    Y = wX + b
    
    Attributes:
        learning_rate (float): Tốc độ học
        max_iterations (int): Số lần lặp tối đa
        tolerance (float): Ngưỡng dừng khi cost không thay đổi
        weights (np.ndarray): Trọng số của model
        bias (float): Bias term
        cost_history (List[float]): Lịch sử cost qua các iteration
        is_fitted (bool): Model đã được train chưa
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 tolerance: float = 1e-6, random_state: Optional[int] = None):
        """
        Initialize Simple Linear Regression
        
        Args:
            learning_rate: Tốc độ học (default: 0.01)
            max_iterations: Số lần lặp tối đa (default: 1000)
            tolerance: Ngưỡng dừng sớm (default: 1e-6)
            random_state: Random seed để reproduce results
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Model parameters
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.is_fitted = False
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_parameters(self, n_features: int) -> None:
        """Initialize weights và bias"""
        # Xavier/Glorot initialization
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
    
    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error"""
        return np.mean((y_true - y_pred) ** 2)
    
    def _compute_gradients(self, X: np.ndarray, y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute gradients cho weights và bias"""
        m = len(y_true)
        
        # Gradient cho weights: dJ/dw = (1/m) * X^T * (y_pred - y_true)
        dw = (1/m) * np.dot(X.T, (y_pred - y_true))
        
        # Gradient cho bias: dJ/db = (1/m) * sum(y_pred - y_true)
        db = (1/m) * np.sum(y_pred - y_true)
        
        return dw, db
    
    def _update_parameters(self, dw: np.ndarray, db: float) -> None:
        """Update parameters using gradient descent"""
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'SimpleLinearRegression':
        """
        Train the model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            verbose: In ra thông tin training
            
        Returns:
            self: Trained model
        """
        # Input validation
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if len(X) != len(y):
            raise ValueError("X và y phải có cùng số lượng samples")
        
        # Initialize parameters
        n_features = X.shape[1]
        self._initialize_parameters(n_features)
        
        # Training loop
        self.cost_history = []
        prev_cost = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward pass (compute prediction directly)
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute cost
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, y_pred)
            
            # Update parameters
            self._update_parameters(dw, db)
            
            prev_cost = cost
            
            # Print progress
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            y_pred: Predicted values (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score
        
        Args:
            X: Feature matrix
            y: True values
            
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_parameters(self) -> dict:
        """Get model parameters"""
        return {
            'weights': self.weights.copy() if self.weights is not None else None,
            'bias': self.bias,
            'cost_history': self.cost_history.copy()
        }


class MultipleLinearRegression:
    """
    Multiple Linear Regression với regularization
    
    Y = w1*X1 + w2*X2 + ... + wn*Xn + b
    
    Hỗ trợ:
    - Ridge Regression (L2 regularization)
    - Lasso Regression (L1 regularization)
    - Elastic Net (L1 + L2 regularization)
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 regularization: str = 'none', alpha: float = 1.0,
                 l1_ratio: float = 0.5, tolerance: float = 1e-6,
                 random_state: Optional[int] = None):
        """
        Initialize Multiple Linear Regression
        
        Args:
            learning_rate: Tốc độ học
            max_iterations: Số lần lặp tối đa
            regularization: Loại regularization ('none', 'ridge', 'lasso', 'elastic_net')
            alpha: Regularization strength
            l1_ratio: Tỷ lệ L1 trong Elastic Net (0 = Ridge, 1 = Lasso)
            tolerance: Ngưỡng dừng sớm
            random_state: Random seed
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Model parameters
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.is_fitted = False
        
        # Validation
        if regularization not in ['none', 'ridge', 'lasso', 'elastic_net']:
            raise ValueError("regularization phải là 'none', 'ridge', 'lasso', hoặc 'elastic_net'")
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_parameters(self, n_features: int) -> None:
        """Initialize parameters"""
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
    
    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute cost với regularization"""
        mse = np.mean((y_true - y_pred) ** 2)
        
        if self.regularization == 'none':
            return mse
        elif self.regularization == 'ridge':
            l2_penalty = self.alpha * np.sum(self.weights ** 2)
            return mse + l2_penalty
        elif self.regularization == 'lasso':
            l1_penalty = self.alpha * np.sum(np.abs(self.weights))
            return mse + l1_penalty
        elif self.regularization == 'elastic_net':
            l1_penalty = self.alpha * self.l1_ratio * np.sum(np.abs(self.weights))
            l2_penalty = self.alpha * (1 - self.l1_ratio) * np.sum(self.weights ** 2)
            return mse + l1_penalty + l2_penalty
    
    def _compute_gradients(self, X: np.ndarray, y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute gradients với regularization"""
        m = len(y_true)
        
        # Base gradients
        dw = (1/m) * np.dot(X.T, (y_pred - y_true))
        db = (1/m) * np.sum(y_pred - y_true)
        
        # Add regularization gradients
        if self.regularization == 'ridge':
            dw += 2 * self.alpha * self.weights
        elif self.regularization == 'lasso':
            dw += self.alpha * np.sign(self.weights)
        elif self.regularization == 'elastic_net':
            dw += self.alpha * self.l1_ratio * np.sign(self.weights)
            dw += 2 * self.alpha * (1 - self.l1_ratio) * self.weights
        
        return dw, db
    
    def _update_parameters(self, dw: np.ndarray, db: float) -> None:
        """Update parameters"""
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'MultipleLinearRegression':
        """Train the model"""
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if len(X) != len(y):
            raise ValueError("X và y phải có cùng số lượng samples")
        
        # Initialize parameters
        n_features = X.shape[1]
        self._initialize_parameters(n_features)
        
        # Training loop
        self.cost_history = []
        prev_cost = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward pass (compute prediction directly)
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute cost
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Check convergence
            if abs(prev_cost - cost) < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, y_pred)
            
            # Update parameters
            self._update_parameters(dw, db)
            
            prev_cost = cost
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_parameters(self) -> dict:
        """Get model parameters"""
        return {
            'weights': self.weights.copy() if self.weights is not None else None,
            'bias': self.bias,
            'cost_history': self.cost_history.copy(),
            'regularization': self.regularization,
            'alpha': self.alpha
        }
