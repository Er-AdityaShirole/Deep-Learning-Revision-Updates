# Lesson 14: Loss Functions in Deep Learning

Welcome to my revision notes for **Lesson 14** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Why we need Loss Functions**: Evaluating model output difference.
2. **Loss Function vs. Cost Function**: Defining terminology.
3. **Regression Losses**: Mean Squared Error (MSE), Mean Absolute Error (MAE), Huber Loss.
4. **Classification Losses**: Binary Cross-Entropy, Categorical Cross-Entropy, Sparse Categorical Cross-Entropy.

---

## 📝 Key Revision Points

### Loss Function vs. Cost Function
- **Loss Function**: Calculated for a **single** training instance.
- **Cost Function**: The **average** loss across the entire training dataset.

---

### 1. Regression Loss Functions

#### Mean Squared Error (MSE) / L2 Loss
$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- **Advantages**: Smooth, differentiable everywhere. Easy to optimize with gradient descent.
- **Disadvantages**: Highly sensitive to outliers due to squaring errors.

#### Mean Absolute Error (MAE) / L1 Loss
$$L = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- **Advantages**: Robust to outliers.
- **Disadvantages**: The derivative is undefined (discontinuous) at $y_i = \hat{y}_i$, making optimization near zero tricky.

#### Huber Loss
Combines MSE and MAE. It acts like MSE for small errors and MAE for large errors:
$$L_{\delta}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
- **Advantages**: Differentiable at zero and robust to outliers.

---

### 2. Classification Loss Functions

#### Binary Cross-Entropy (BCE) / Log Loss
Used for binary classification tasks (outputs are $0$ or $1$):
$$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

#### Categorical Cross-Entropy (CCE)
Used for multi-class classification where labels are **one-hot encoded**:
$$L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

#### Sparse Categorical Cross-Entropy
Used for multi-class classification where labels are **integers** (e.g. `y = 3` instead of `y = [0, 0, 0, 1]`). Computes the exact same math as CCE but saves memory.
