# Lesson 32: Optimizers in Deep Learning

Welcome to my revision notes for **Lesson 32** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Role of Optimizers**: Updating weights to reduce the loss function.
2. **Gradient Descent Challenges**: Learning rate, local minima, saddle points, and plateaus.
3. **Advanced Optimizers**: Motivation for momentum, AdaGrad, RMSprop, Adam, and NAG.

---

## Key Revision Points

### What Optimizers Do
An optimizer decides how model parameters should change after computing gradients.

Basic gradient descent update:

$$w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}$$

Here `w` is a weight, `L` is the loss, and `eta` is the learning rate.

### Learning Rate Problem
The learning rate controls the size of each update.

If it is too small, training becomes very slow. If it is too large, the model may overshoot the minimum and fail to converge.

### Local Minima and Saddle Points
Deep learning loss surfaces are usually non-convex. Training can face:

- **local minima**, where the model gets stuck in a smaller valley,
- **saddle points**, where gradients can become very small,
- **plateaus**, where progress becomes slow.

### High Curvature Problem
In narrow valleys, normal gradient descent can zig-zag across the slope instead of moving smoothly toward the minimum.

This wastes updates and slows convergence.

### Why Advanced Optimizers Are Needed
Advanced optimizers improve the update direction or learning rate behavior.

- **Momentum** uses past gradients to smooth updates.
- **AdaGrad** adapts learning rates for each parameter.
- **RMSprop** improves AdaGrad using moving averages.
- **Adam** combines momentum and adaptive learning rates.
- **NAG** looks ahead before computing the correction.

### Main Takeaway
Optimizers are not just implementation details. They strongly affect training speed, stability, and the ability to handle difficult loss surfaces.
