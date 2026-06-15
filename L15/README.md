# Lesson 15: Backpropagation - Part 1 (The What?)

Welcome to my revision notes for **Lesson 15** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **The Core Concept of Backpropagation**: How networks learn.
2. **Forward Pass vs. Backward Pass**: The directional flow of information.
3. **Weight Update Mechanics**: The role of gradients.

---

## 📝 Key Revision Points

### What is Backpropagation?
Backpropagation is the core algorithm used to train neural networks. It stands for **"backward propagation of errors"**.

```
[Forward Pass: Predicts output] --> [Calculate Loss] --> [Backward Pass: Calculate Gradients] --> [Update Weights]
```

### The Three Core Steps of Training:
1. **Forward Pass**: Feed inputs through the network, calculating activations layer-by-layer, to produce a prediction $\hat{y}$.
2. **Calculate Loss**: Compare predictions $\hat{y}$ with true labels $y$ using a loss function (e.g., MSE or Cross-Entropy).
3. **Backward Pass (Backpropagation)**: Apply the chain rule of calculus to propagate the loss backward, computing the partial derivatives (gradients) of the loss function with respect to every weight and bias:
   $$\frac{\partial L}{\partial W}, \quad \frac{\partial L}{\partial b}$$
4. **Update Weights**: Adjust the weights using an optimization algorithm (e.g., Gradient Descent):
   $$W_{\text{new}} = W_{\text{old}} - \alpha \frac{\partial L}{\partial W}$$
   $$b_{\text{new}} = b_{\text{old}} - \alpha \frac{\partial L}{\partial b}$$
