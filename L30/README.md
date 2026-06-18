# Lesson 30: Xavier and He Weight Initialization

Welcome to my revision notes for **Lesson 30** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Better Random Initialization**: Choosing random weights with the right scale.
2. **Xavier/Glorot Initialization**: Useful for sigmoid and tanh based networks.
3. **He Initialization**: Useful for ReLU based networks.

---

## Key Revision Points

### Problem with Plain Random Initialization
Random weights break symmetry, but their scale still matters.

If weights are too large, activations and gradients may explode. If weights are too small, signals may shrink layer by layer and cause vanishing gradients.

### Xavier/Glorot Initialization
Xavier initialization tries to keep the variance of activations stable across layers.

For a layer with `fan_in` input connections and `fan_out` output connections:

$$Var(W) = \frac{2}{fan\_in + fan\_out}$$

Common normal form:

$$W \sim N\left(0, \sqrt{\frac{2}{fan\_in + fan\_out}}\right)$$

It works well with activations like **sigmoid** and **tanh**, where keeping values in a stable range is important.

### He Initialization
He initialization is designed for ReLU and its variants.

Since ReLU blocks negative values, the initialization uses a slightly larger variance:

$$Var(W) = \frac{2}{fan\_in}$$

Common normal form:

$$W \sim N\left(0, \sqrt{\frac{2}{fan\_in}}\right)$$

### When to Use What

- Use **Xavier/Glorot** with sigmoid or tanh.
- Use **He** with ReLU, Leaky ReLU, or similar activations.

### Main Takeaway
Good initialization keeps signals flowing through the network. Xavier and He initialization are not random guesses; they scale weights based on the number of connections in each layer.
