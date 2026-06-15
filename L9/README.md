# Lesson 9: Multi-layer Perceptron (MLP) Intuition

Welcome to my revision notes for **Lesson 9** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Perceptron with Sigmoid**: Replacing step functions with continuous probabilities.
2. **How Nodes and Hidden Layers Warp Space**: Intuitive geometric understanding of MLP operations.
3. **TensorFlow Playground Demo**: Observing neural networks learn complex decision boundaries.

---

## 📝 Key Revision Points

### The Continuous Neuron: Perceptron with Sigmoid
Standard perceptrons use step functions, which output strictly $0$ or $1$. This makes differentiation impossible. By replacing the step function with the **Sigmoid function**, we get continuous outputs in the range $(0, 1)$ representing probabilities:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

$$\hat{y} = \sigma(W \cdot X + b)$$

### Architecture Intuition
- **Adding nodes to a hidden layer**: Increases the number of decision lines, allowing the network to form complex polygonal boundaries in the feature space.
- **Adding hidden layers**: Creates hierarchical abstractions. Layer 1 detects simple edges/combinations, Layer 2 combines these to detect shapes, Layer 3 detects objects.
- **Activation Functions**: Essential to introduce non-linearity. Without them, a multi-layer neural network collapses into a single linear model, regardless of how many layers you add.

### TensorFlow Playground Experiment Summary
Using Sigmoid vs. ReLU in TensorFlow Playground:
- Sigmoid activation can take a high number of epochs (e.g., $600+$) to fit non-linear distributions.
- ReLU activation converges much faster (e.g., $60-100$ epochs) because its gradient does not saturate for positive inputs.
