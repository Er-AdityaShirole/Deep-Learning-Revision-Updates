# Lesson 16: Backpropagation - Part 2 (The How - Math & Regression Code)

Welcome to my revision notes for **Lesson 16** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Mathematical Derivation**: Using the Chain Rule to calculate gradients.
2. **Backpropagation Code Walkthrough**: Implementing forward and backward passes from scratch in Python.

---

## 📝 Mathematical Intuition & Chain Rule

To understand how a change in weight $w_{11}^{(1)}$ affects the overall loss $L$, we use the chain rule of calculus:

$$\frac{\partial L}{\partial w_{11}^{(1)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial a_{1}^{(1)}} \cdot \frac{\partial a_{1}^{(1)}}{\partial z_{1}^{(1)}} \cdot \frac{\partial z_{1}^{(1)}}{\partial w_{11}^{(1)}}$$

By multiplying these partial derivatives, the error is effectively passed backward, layer-by-layer, computing exact gradient updates.

---

## 💻 Code Walkthrough (from `backpropagation-regression.ipynb`)

In this lesson, we build a simple regression neural network with a hidden layer and train it manually.

### Parameter Initialization:
```python
def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters
```

### Forward Pass:
```python
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    return Z
```

### Manual Gradient Update Step (Gradient Descent Loop):
For each training example, we compute predicted `y_hat`, calculate gradients manually using derivatives, and adjust parameters:
```python
# Weight Update Formulas implemented in the notebook loop:
# W2 = W2 - lr * dL/dW2
# b2 = b2 - lr * dL/db2
# W1 = W1 - lr * dL/dW1
# b1 = b1 - lr * dL/db1
```
The manual loss decreases steadily epoch-by-epoch (e.g., from $25.32$ down to $1.30$), proving that backpropagation works mathematically!
