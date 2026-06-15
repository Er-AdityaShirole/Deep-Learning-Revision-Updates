# Lesson 5: The Perceptron Trick (Training Algorithm)

Welcome to my revision notes for **Lesson 5** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **How to Train a Perceptron**: The Perceptron Trick.
2. **Mathematical Derivation of the Update Rule**: Adjusting the decision boundary based on misclassified points.
3. **Learning Rate ($\\eta$)**: Controlling step sizes during optimization.

---

## 📝 Key Revision Points

### The Perceptron Trick
The perceptron trick is a heuristic method to update the weights and bias when a training instance is misclassified.

> [!NOTE]
> If a point is **correctly classified**, we make **no changes** to the weights or bias.
> If a point is **misclassified**, we update the weights and bias to move the decision boundary closer to the misclassified point.

```
Decision Boundary: w_1 x_1 + w_2 x_2 + b = 0
```

#### Update Rules for Misclassified Points:
Let $\eta$ be the learning rate (usually a small value like $0.01$ or $0.1$), $X_i = (x_{i1}, x_{i2})$ be the input vector, $y_i$ be the true label ($0$ or $1$), and $\hat{y}_i$ be the predicted label.

- **Case 1: True label $y_i = 1$, but Predicted $\hat{y}_i = 0$** (Line needs to move towards the point):
  $$w_{j} \leftarrow w_{j} + \eta x_{ij}$$
  $$b \leftarrow b + \eta$$

- **Case 2: True label $y_i = 0$, but Predicted $\hat{y}_i = 1$** (Line needs to move away from the point):
  $$w_{j} \leftarrow w_{j} - \eta x_{ij}$$
  $$b \leftarrow b - \eta$$

### Combined General Rule:
Using the error $e_i = (y_i - \hat{y}_i)$, we can write the update rule as:
$$w_j \leftarrow w_j + \eta (y_i - \hat{y}_i) x_{ij}$$
$$b \leftarrow b + \eta (y_i - \hat{y}_i)$$

- If $y_i = \hat{y}_i$, the term $(y_i - \hat{y}_i) = 0$, resulting in no change (as expected).
