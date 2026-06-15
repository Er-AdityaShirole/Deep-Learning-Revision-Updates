# Lesson 6: Perceptron Loss Function & Gradient Descent

Welcome to my revision notes for **Lesson 6** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Why the Perceptron Trick is Heuristic**: The need for a formal loss function.
2. **Formulating the Perceptron Loss Function**: Mathematical optimization view.
3. **Gradient Descent Optimization**: How we update weights using derivatives.
4. **Convex vs. Non-Convex Optimization**: Local vs. Global Minima in Deep Learning.

---

## 📝 Key Revision Points

### The Problem with the Perceptron Trick
The perceptron trick adjusts the line step-by-step but lacks a global mathematical function that evaluates *how bad* the model is across the entire dataset. It only seeks *any* separating boundary rather than the *best* boundary.

### Perceptron Loss Function
To optimize the perceptron formally, we define a loss function that measures the distance of misclassified points from the decision boundary.

For a training sample $(x_i, y_i)$ where $y_i \in \{-1, +1\}$:
$$L(w, b) = \sum_{i \in M} -y_i(w \cdot x_i + b)$$
where $M$ is the set of all misclassified points.
- If a point is misclassified, $y_i(w \cdot x_i + b) < 0$, so the term $-y_i(w \cdot x_i + b)$ is positive.
- If it is correctly classified, its loss is $0$.

### Gradient Descent Optimization
To minimize the Loss $L(w, b)$, we take steps in the opposite direction of the gradient:
$$w \leftarrow w - \eta \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \eta \frac{\partial L}{\partial b}$$

Calculating the derivatives for a single misclassified point:
$$\frac{\partial L}{\partial w} = -y_i x_i \implies w \leftarrow w + \eta y_i x_i$$
$$\frac{\partial L}{\partial b} = -y_i \implies b \leftarrow b + \eta y_i$$
This derivative derivation perfectly mathematically justifies the heuristic Perceptron Trick!

### Convex vs. Non-Convex Problems
> [!IMPORTANT]
> - **Convex Problems** (e.g., Linear Regression, Logistic Regression): The loss function looks like a giant bowl with **only one minimum** (global minimum). Gradient descent is guaranteed to find it.
> - **Non-Convex Problems** (e.g., Deep Neural Networks): The loss surface has many hills, valleys, and **local minima**. Gradient descent is not guaranteed to find the absolute global optimum, but random initialization and optimization tweaks usually find a solution that is "good enough" for practical use.
