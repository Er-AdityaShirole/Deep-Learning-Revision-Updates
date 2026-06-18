# Lesson 35: Nesterov Accelerated Gradient

Welcome to my revision notes for **Lesson 35** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Nesterov Accelerated Gradient**: A smarter version of momentum.
2. **Look-Ahead Update**: Estimating where momentum will take the weights.
3. **Momentum vs NAG**: Correcting the update before overshooting.

---

## Key Revision Points

### Problem with Momentum
Momentum speeds up learning, but it can overshoot the minimum because it keeps moving in the accumulated direction.

This can create oscillations near the minimum.

### Nesterov Intuition
Nesterov Accelerated Gradient first looks ahead in the direction of the current momentum, then calculates the gradient at that future position.

This gives the optimizer a chance to correct its direction earlier.

### Standard Momentum
Momentum updates using the gradient at the current position:

$$v_t = \beta v_{t-1} + \eta \nabla_w L(w_t)$$

$$w_{t+1} = w_t - v_t$$

### Nesterov Accelerated Gradient
NAG computes the gradient after a look-ahead step:

$$v_t = \beta v_{t-1} + \eta \nabla_w L(w_t - \beta v_{t-1})$$

$$w_{t+1} = w_t - v_t$$

### Why NAG Helps
NAG can slow down earlier when it is moving too fast toward a minimum.

Compared with normal momentum, it often gives a more informed and controlled update direction.

### Main Takeaway
NAG improves momentum by looking ahead before applying the gradient correction. It keeps the speed benefits of momentum while reducing overshooting.
