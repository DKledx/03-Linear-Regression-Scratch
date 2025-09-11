# 📈 Build Your Own Linear Regression

## 🎯 Mục Tiêu Dự Án

Dự án **Linear Regression from Scratch** sẽ dạy bạn hiểu sâu về thuật toán hồi quy tuyến tính bằng cách tự implement từ đầu. Bạn sẽ code gradient descent, cost function, và toàn bộ quá trình training mà không dùng thư viện có sẵn. Đây là nền tảng quan trọng để hiểu machine learning algorithms.

## 🎓 Kiến Thức Sẽ Học Được

### 📚 Mathematical Foundations
- **Linear Algebra**: Vector operations, matrix multiplication
- **Calculus**: Derivatives, partial derivatives
- **Statistics**: Mean, variance, correlation
- **Optimization**: Gradient descent algorithm

### 🔧 Core Concepts
- **Linear Regression**: Hồi quy tuyến tính
- **Cost Function**: Hàm mất mát (MSE, MAE)
- **Gradient Descent**: Thuật toán tối ưu
- **Learning Rate**: Tốc độ học
- **Feature Scaling**: Chuẩn hóa dữ liệu

### 🛠️ Implementation Skills
- **NumPy**: Numerical computing
- **Vectorization**: Tối ưu hóa tính toán
- **Object-Oriented Programming**: Class design
- **Debugging**: Tìm và sửa lỗi

## 📁 Cấu Trúc Dự Án

```
03-Linear-Regression-Scratch/
├── README.md
├── notebooks/
│   ├── 01-mathematical-foundations.ipynb
│   ├── 02-simple-linear-regression.ipynb
│   ├── 03-multiple-linear-regression.ipynb
│   ├── 04-gradient-descent-deep-dive.ipynb
│   ├── 05-regularization.ipynb
│   └── 06-comparison-with-sklearn.ipynb
├── src/
│   ├── linear_regression.py
│   ├── gradient_descent.py
│   ├── cost_functions.py
│   ├── data_generator.py
│   └── visualization.py
├── data/
│   ├── synthetic/
│   └── real_world/
├── tests/
│   ├── test_linear_regression.py
│   └── test_gradient_descent.py
├── requirements.txt
└── .gitignore
```

## 📊 Dataset Overview

### 🎲 Synthetic Data
- **Linear Data**: Y = 2X + 3 + noise
- **Multiple Features**: Y = 3X1 + 2X2 - X3 + 5 + noise
- **Non-linear Data**: Y = X² + 2X + 1 + noise (để test limitations)

### 🏠 Real-world Data
- **Housing Dataset**: sklearn.datasets.fetch_california_housing
- **Boston Housing**: sklearn.datasets.load_boston (deprecated)
- **Custom Dataset**: Tạo từ Kaggle

## 🚀 Cách Bắt Đầu

### 1. Cài Đặt Môi Trường
```bash
# Tạo virtual environment
python -m venv lr_env
source lr_env/bin/activate  # Linux/Mac
# hoặc
lr_env\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chạy Jupyter Notebook
```bash
jupyter notebook
```

### 3. Bắt Đầu với Mathematical Foundations
Mở `notebooks/01-mathematical-foundations.ipynb` để hiểu toán học đằng sau!

### 4. Chạy test để xác nhận implementation
```bash
pytest -q
```

## 📋 Roadmap Học Tập

### ✅ Phase 1: Mathematical Foundations
- [ ] Linear algebra basics
- [ ] Calculus for machine learning
- [ ] Cost functions (MSE, MAE)
- [ ] Gradient descent intuition

> Lưu ý: Bạn có thể tick các mục này sau khi hoàn thành từng notebook tương ứng.

### ✅ Phase 2: Simple Linear Regression
- [ ] Implement from scratch
- [ ] Visualize cost function
- [ ] Gradient descent implementation
- [ ] Convergence analysis

### ✅ Phase 3: Multiple Linear Regression
- [ ] Vectorized implementation
- [ ] Feature scaling
- [ ] Normal equation
- [ ] Performance comparison

### ✅ Phase 4: Advanced Topics
- [ ] Regularization (Ridge, Lasso)
- [ ] Learning rate scheduling
- [ ] Batch vs Stochastic gradient descent
- [ ] Feature engineering

### ✅ Phase 5: Validation & Testing
- [ ] Unit tests
- [ ] Comparison with sklearn
- [ ] Real-world applications
- [ ] Performance optimization

## 🧮 Mathematical Implementation

### 📐 Simple Linear Regression
```python
class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Compute cost
            cost = self.compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1/len(X)) * np.dot(X.T, (y_pred - y))
            db = (1/len(X)) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def compute_cost(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
```

> Gợi ý thực hành: thử thay đổi `learning_rate`, `max_iterations` và quan sát đường cong học `cost_history`.

### 📊 Cost Function Visualization
```python
def plot_cost_function(X, y, weight_range, bias_range):
    """Visualize cost function in 3D"""
    W, B = np.meshgrid(weight_range, bias_range)
    costs = np.zeros_like(W)
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            y_pred = W[i,j] * X + B[i,j]
            costs[i,j] = np.mean((y - y_pred) ** 2)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(W, B, costs, alpha=0.7, cmap='viridis')
    ax1.set_xlabel('Weight')
    ax1.set_ylabel('Bias')
    ax1.set_zlabel('Cost')
    ax1.set_title('Cost Function Surface')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(W, B, costs, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('Weight')
    ax2.set_ylabel('Bias')
    ax2.set_title('Cost Function Contours')
    
    plt.tight_layout()
    plt.show()
```

## 🎯 Gradient Descent Variants

### 📈 Batch Gradient Descent
```python
def batch_gradient_descent(X, y, learning_rate=0.01, max_iterations=1000):
    """Standard gradient descent using all data"""
    m = len(y)
    theta = np.random.normal(0, 0.01, X.shape[1])
    cost_history = []
    
    for i in range(max_iterations):
        # Compute predictions
        y_pred = np.dot(X, theta)
        
        # Compute cost
        cost = np.mean((y - y_pred) ** 2)
        cost_history.append(cost)
        
        # Compute gradient
        gradient = (1/m) * np.dot(X.T, (y_pred - y))
        
        # Update parameters
        theta -= learning_rate * gradient
    
    return theta, cost_history
```

### ⚡ Stochastic Gradient Descent
```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, max_epochs=100):
    """SGD using one sample at a time"""
    m = len(y)
    theta = np.random.normal(0, 0.01, X.shape[1])
    cost_history = []
    
    for epoch in range(max_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_cost = 0
        for i in range(m):
            # Single sample
            x_i = X_shuffled[i:i+1]
            y_i = y_shuffled[i:i+1]
            
            # Compute prediction
            y_pred = np.dot(x_i, theta)
            
            # Compute cost
            cost = (y_i - y_pred) ** 2
            epoch_cost += cost[0]
            
            # Compute gradient
            gradient = np.dot(x_i.T, (y_pred - y_i))
            
            # Update parameters
            theta -= learning_rate * gradient
        
        cost_history.append(epoch_cost / m)
    
    return theta, cost_history
```

## 📊 Evaluation & Visualization

### 🎯 Model Performance
```python
def evaluate_model(y_true, y_pred):
    """Comprehensive model evaluation"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
```

### 🔎 Mẹo đánh giá nhanh
- **MSE/RMSE thấp**: mô hình khớp tốt dữ liệu.
- **R² gần 1**: mô hình giải thích tốt phương sai của dữ liệu.

### 📈 Learning Curves
```python
def plot_learning_curves(cost_history, title="Learning Curve"):
    """Plot cost function over iterations"""
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()
```

## 🔧 Advanced Features

### 🎛️ Regularization
```python
class RidgeRegression:
    def __init__(self, alpha=1.0, learning_rate=0.01):
        self.alpha = alpha  # Regularization parameter
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Ridge regression with L2 regularization
        m = len(y)
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        self.bias = 0
        
        for i in range(1000):
            y_pred = self.predict(X)
            
            # Cost with regularization
            cost = np.mean((y - y_pred) ** 2) + self.alpha * np.sum(self.weights ** 2)
            
            # Gradients with regularization
            dw = (1/m) * np.dot(X.T, (y_pred - y)) + 2 * self.alpha * self.weights
            db = (1/m) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
```

> Thực hành: thay đổi `alpha` để thấy tác động của regularization lên trọng số.

## 📝 Notes Thực Thi Dự Án

- **Quy ước môi trường**: dùng `lr_env` cho riêng dự án này.
- **Chạy test trước commit**: đảm bảo `pytest` pass 100%.
- **Kiểm thử thủ công**: chạy notebook `02-simple-linear-regression` để xem đường hồi quy và learning curve.
- **Quy tắc dữ liệu**: dữ liệu synthetic nằm ở `data/synthetic/`, dữ liệu thật ở `data/real_world/`.
- **Quản lý phụ thuộc**: cập nhật `requirements.txt` khi thêm thư viện mới.
- **Định dạng code**: có thể dùng `black` và `flake8` (đã khai báo trong requirements).

## 🧪 Hướng Dẫn Test Nhanh

```bash
python - << 'PY'
import numpy as np
from src.linear_regression import SimpleLinearRegression

X = np.linspace(-5,5,200).reshape(-1,1)
y = 2*X.squeeze() + 3 + np.random.normal(0,0.2,200)

model = SimpleLinearRegression(learning_rate=0.05, max_iterations=5000, tolerance=1e-10, random_state=42)
model.fit(X,y)
print(model.get_parameters())
print('R2=', model.score(X,y))
PY
```

## 🎯 Expected Results

Sau khi hoàn thành dự án này, bạn sẽ có:

1. **Deep Understanding**: Hiểu sâu về linear regression
2. **Implementation Skills**: Tự code được thuật toán
3. **Mathematical Intuition**: Hiểu toán học đằng sau
4. **Optimization Knowledge**: Gradient descent variants
5. **Performance Comparison**: So sánh với sklearn

## 🔍 Key Insights to Discover

### 📊 Mathematical Insights
- **Cost function shape** ảnh hưởng đến convergence
- **Learning rate** quyết định tốc độ học
- **Feature scaling** quan trọng cho gradient descent
- **Regularization** giúp tránh overfitting

### 🎯 Implementation Insights
- **Vectorization** tăng tốc đáng kể
- **Batch size** ảnh hưởng đến stability
- **Initialization** quan trọng cho convergence
- **Early stopping** tránh overfitting

## 📚 Tài Liệu Tham Khảo

- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [Linear Regression Mathematics](https://en.wikipedia.org/wiki/Linear_regression)
- [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
- [NumPy Documentation](https://numpy.org/doc/)

## 🏆 Next Steps

Sau khi hoàn thành Linear Regression from Scratch, bạn có thể:
- Implement other algorithms from scratch
- Chuyển sang dự án 4: Titanic Survival Prediction
- Thử advanced optimization techniques
- Phát triển mini ML library

## 🎨 Bonus: Interactive Demo

Tạo Streamlit app để demo:
- Adjust learning rate và iterations
- Visualize gradient descent in real-time
- Compare different algorithms
- Upload custom datasets

---

**Happy Coding! 📈**

*Hãy bắt đầu với mathematical foundations và xây dựng linear regression từ con số 0!*
