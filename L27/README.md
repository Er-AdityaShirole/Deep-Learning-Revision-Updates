# Lesson 27: Activation Functions

Welcome to my revision notes for **Lesson 27** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **What Activation Functions Do**: Transforming the weighted sum of inputs into neuron output.
2. **Sigmoid, Tanh, and ReLU**: Comparing common activation functions.
3. **Advantages and Disadvantages**: Saturation, zero-centering, computation cost, and gradient flow.

---

## Key Revision Points

### What is an Activation Function?
Each neuron first computes a weighted sum:

$$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

Then an activation function transforms it:

$$a = g(z)$$

Without activation functions, a neural network would behave like a linear model even if it had many layers.

### Sigmoid
Maps values between `0` and `1`:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Useful for binary classification output layers, but it can saturate and cause vanishing gradients.

### Tanh
Maps values between `-1` and `1`, so it is zero-centered:

$$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

It is often better than sigmoid in hidden layers, but still has saturation problems.

### ReLU
Returns `0` for negative inputs and `x` for positive inputs:

$$ReLU(x) = max(0, x)$$

Advantages:
- Simple and computationally cheap.
- Does not saturate in the positive region.
- Helps reduce vanishing gradient problems.
- Usually converges faster than sigmoid/tanh.

Disadvantages:
- Not zero-centered.
- Gradient is `0` for negative values, which can lead to dead neurons.
