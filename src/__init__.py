"""
Linear Regression from Scratch

Một thư viện đơn giản để implement Linear Regression từ đầu,
giúp hiểu sâu về thuật toán và toán học đằng sau.

Author: ML Portfolio Project
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "ML Portfolio Project"

# Import main classes
from .linear_regression import SimpleLinearRegression, MultipleLinearRegression
from .gradient_descent import GradientDescent, StochasticGradientDescent
from .cost_functions import MeanSquaredError, MeanAbsoluteError
from .data_generator import DataGenerator
from .visualization import Plotter

__all__ = [
    'SimpleLinearRegression',
    'MultipleLinearRegression', 
    'GradientDescent',
    'StochasticGradientDescent',
    'MeanSquaredError',
    'MeanAbsoluteError',
    'DataGenerator',
    'Plotter'
]
