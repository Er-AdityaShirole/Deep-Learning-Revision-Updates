# Lesson 19: MLP Memoization

Welcome to my revision notes for **Lesson 19** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Memoization in Backpropagation**: Storing values from forward propagation so they can be reused during backward propagation.
2. **Why Memoization Matters**: Avoiding repeated calculations for the same neuron outputs and derivatives.
3. **Backprop Storage View**: Keeping track of activations, weights, biases, and derivatives layer by layer.

---

## Key Revision Points

### What is Memoization?
Memoization means saving the result of an expensive calculation and reusing it when the same value is needed again.

In neural networks, forward propagation already computes values like:

$$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = g(z^{(l)})$$

During backpropagation, these same activations and derivatives are needed again. Instead of recalculating everything, we store them.

### Why it is Useful in MLPs
- Backpropagation needs intermediate values from every layer.
- Derivatives such as $g'(z)$ depend on previously computed $z$ values.
- Storing values makes the backward pass faster and cleaner.
- It also helps organize the chain rule calculation across many layers.

### Main Idea
Forward propagation is not only used for prediction. It also prepares the cache required by backpropagation:

```text
Forward pass  -> compute and store activations
Backward pass -> reuse stored values to compute gradients
```

This is why memoization is closely connected to efficient training in deep neural networks.
