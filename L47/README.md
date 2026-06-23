# Lesson 47: Backpropagation in CNN: Part 1 (Derivatives & Dense-Conv boundaries)

Welcome to my revision notes for **Lesson 47** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **The Math of Backpropagation in CNNs**: Expanding the chain rule.
2. **Dense to Conv Boundary**: Propagating loss gradients backwards from the Flatten layer into the preceding MaxPooling/Conv layers.
3. **Calculating Gradients for Weights and Biases**: Analytical derivatives.

---

## 📝 Mathematical Intuition & Chain Rule

Backpropagation in a CNN follows the same fundamental principles of calculus as an ANN, but we must account for the shared weights and spatial pooling operations.

```
Loss (L) ---> Dense Output ---> Flatten ---> MaxPooling ---> Conv Layers ---> Input
```

### Deriving the Gradients at the Flatten Layer Boundary:
Let $x$ be the output of the final convolutional/maxpooling layer, which is flattened into a 1D vector $a = \text{flatten}(x)$ before entering the dense layers.
During the backward pass, we receive the gradient of the loss with respect to the flattened activations:
$$\frac{\partial L}{\partial a_i}$$

We reshape these 1D gradients back into the original 2D/3D spatial shape of the layer preceding the flatten layer:
$$\frac{\partial L}{\partial x_{r, c, d}} = \text{reshape}\left(\frac{\partial L}{\partial a_i}\right)$$

This allows us to continue passing the error gradient backward through the spatial layers.

### Weight Gradients in Convolutional Layers:
For a kernel filter weight $w_{a, b}$ convolving over input $X$:
$$\frac{\partial L}{\partial w_{a, b}} = \sum_{i} \sum_{j} \frac{\partial L}{\partial Z_{i, j}} X_{i+a, j+b}$$
This means the gradient with respect to a shared weight is the sum of the gradients across all positions where that weight was applied!
