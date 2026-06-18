# Lesson 34: SGD with Momentum

Welcome to my revision notes for **Lesson 34** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Problem with Normal SGD**: Zig-zag movement and slow convergence.
2. **Momentum Optimization**: Using velocity from previous gradients.
3. **Effect of Beta**: Controlling how strongly past gradients influence the update.

---

## Key Revision Points

### Problem with SGD
In curved loss surfaces, normal SGD can oscillate heavily across the steep direction.

This is common in narrow valleys, where one direction has high curvature and another direction has low curvature.

### Momentum Intuition
Momentum works like adding velocity to gradient descent.

Instead of using only the current gradient, it combines the current gradient with the previous update direction.

### Momentum Equations
Velocity update:

$$v_t = \beta v_{t-1} + \eta \nabla_w L$$

Weight update:

$$w_t = w_{t-1} - v_t$$

Here `v_t` is velocity, `beta` is the momentum coefficient, `eta` is the learning rate, and `nabla_w L` is the gradient of the loss.

### Why Momentum Helps
Momentum reduces unnecessary zig-zag movement and builds speed in the consistent descent direction.

It is especially useful when the loss surface has high curvature.

### Effect of Beta
Common values are around `0.9`.

- Small `beta`: less memory, behaves closer to normal SGD.
- Large `beta`: smoother movement, but too much momentum can overshoot.

### Main Takeaway
Momentum makes gradient descent more stable and faster by remembering the direction of previous updates.
