# Lesson 18: The Vanishing Gradient Problem

Welcome to my revision notes for **Lesson 18** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **What is the Vanishing Gradient Problem?**: Why deep neural networks stop learning.
2. **Causes**: Deeper architectures combined with saturating activation functions (Sigmoid, Tanh).
3. **Solutions**: ReLU, proper weight initializations, Batch Normalization, and Residual connections.

---

## 📝 Key Revision Points

### What is Vanishing Gradient?
During backpropagation, gradients are computed using chain rule multiplications. In deep networks, as we multiply many terms from the output layer back to the input layer, if these terms are small (less than 1), the gradient shrinks exponentially.
By the time the gradient reaches the early layers, it becomes vanishingly close to $0$:

$$\frac{\partial L}{\partial W_{\text{early}}} \approx 0$$

As a result, the early layers update their weights very slowly or not at all, preventing the network from learning basic features.

```
Input Layer -------- [Early Layers: Gradient ≈ 0 (No Learning)] -------- [Deep Layers: Learning] -------- Output
```

### Major Causes
1. **Activation Functions**: Functions like **Sigmoid** have derivatives that max out at only $0.25$. **Tanh** has derivatives that max out at $1.0$, but saturate (flat slope) for large inputs.
2. **Network Depth**: More layers mean more multiplications in the chain rule, amplifying the shrinking effect.

### Solutions
- **ReLU (Rectified Linear Unit)**: The derivative of ReLU is $1.0$ for all positive inputs. Gradients pass through unchanged, solving the vanishing gradient issue:
  $$f(x) = \max(0, x) \implies f'(x) = 1 \quad (\text{for } x > 0)$$
- **Proper Weight Initialization**: Using techniques like **He Initialization** (for ReLU) or **Xavier/Glorot Initialization** (for Sigmoid/Tanh) to keep weights in a stable scale.
- **Batch Normalization**: Normalizing layer inputs to prevent activations from sliding into flat saturation regions.
- **Residual Connections (Skip Connections)**: Used in ResNets. They allow gradients to bypass layers and flow directly backwards:
  $$a^{(l)} = g(z^{(l)} + a^{(l-1)})$$
