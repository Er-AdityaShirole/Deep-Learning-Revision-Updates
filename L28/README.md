# Lesson 28: ReLU Variants

Welcome to my revision notes for **Lesson 28** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Dying ReLU Problem**: When ReLU neurons output zero permanently.
2. **Leaky ReLU**: Allowing a small negative slope instead of hard zero.
3. **Parametric ReLU, ELU, and SELU**: Variants designed to improve gradient flow.

---

## Key Revision Points

### Dying ReLU Problem
ReLU is excellent for hidden layers, but for negative inputs its output and derivative are both zero:

$$ReLU(x) = max(0, x)$$

If a neuron keeps receiving negative inputs, it may stop updating because its gradient becomes zero. This is called a **dead neuron**.

### Leaky ReLU
Leaky ReLU fixes this by allowing a small slope for negative values:

$$f(x) = max(\alpha x, x)$$

Usually $\alpha$ is a small value such as `0.01`.

### Parametric ReLU
PReLU is similar to Leaky ReLU, but the negative slope is learned during training instead of being fixed.

### ELU
ELU gives smoother negative outputs and can help activations stay closer to zero-centered.

### SELU
SELU is designed for self-normalizing neural networks and works best under specific conditions such as suitable initialization and architecture choices.

### Main Takeaway
ReLU is the default hidden-layer choice, but its variants are useful when dead neurons or poor gradient flow become a problem.
