# Lesson 60: Problems with RNNs (Vanishing & Exploding Gradients)

Welcome to my revision notes for **Lesson 60** of the *100 Days of Deep Learning* course by CampusX. In this lesson, we wrap up the core study of standard Recurrent Neural Networks (SimpleRNNs) by exploring their main limitations during training and why they struggle with long sequences.

---

## 📚 Topics Covered

1. **The Long-Term Dependency Problem**: Why standard RNNs struggle to remember information over long sequences.
2. **Mathematical Derivation of the Gradient Problem**: Expanding Backpropagation Through Time (BPTT) chain rule.
3. **Vanishing Gradients in RNNs**: Why early time steps stop learning.
4. **Exploding Gradients in RNNs**: What causes weight instability and `NaN` values.
5. **Solutions & Mitigations**: Gradient clipping, weight initialization, and introduction to LSTMs/GRUs.

---

## 📝 Key Revision Points

### 1. The Long-Term Dependency Problem
An RNN is designed to maintain a hidden state $h_t$ that carries information from past time steps. However, as the sequence length ($T$) grows, the influence of early inputs ($x_1, x_2, \dots$) on the current output ($y_T$) decays dramatically.

For example, in a text prediction task:
- *"The **cat**, which ate the fish, chased the mouse, and ran up the tree, was **hungry**."* (The model needs to connect the singular subject "**cat**" to the verb "**was**" across many intermediate words).
- In standard RNNs, by the time the network reaches the end of a long sentence, the hidden state $h_t$ has effectively "forgotten" the initial words.

---

### 2. The Mathematics of Backpropagation Through Time (BPTT)

Let's analyze why this memory loss happens by looking at the gradients.
For a standard RNN cell, the hidden state at time step $t$ is:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

If we compute the loss at the final time step $T$ as $L_T$, we want to update the recurrent weights $W_{hh}$. The gradient is computed using BPTT:
$$\frac{\partial L_T}{\partial W_{hh}} = \sum_{k=1}^T \frac{\partial L_T}{\partial h_T} \frac{\partial h_T}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}$$

Let's zoom in on the middle term, $\frac{\partial h_T}{\partial h_k}$, which represents how much the hidden state at time step $T$ changes with respect to the hidden state at time step $k$:
$$\frac{\partial h_T}{\partial h_k} = \prod_{j=k+1}^T \frac{\partial h_j}{\partial h_{j-1}}$$

For any step $j$, the derivative of $h_j$ with respect to $h_{j-1}$ is:
$$\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}\left(1 - \tanh^2(z_j)\right) W_{hh}^T$$
where $z_j = W_{hh} h_{j-1} + W_{xh} x_j + b_h$.

Thus, the product becomes:
$$\frac{\partial h_T}{\partial h_k} = \prod_{j=k+1}^T \text{diag}\left(1 - \tanh^2(z_j)\right) W_{hh}^T$$

> [!IMPORTANT]
> Since we are multiplying the transpose of the recurrent weight matrix $W_{hh}$ and the diagonal matrix of the activation derivative repeatedly $(T - k)$ times, the magnitude of the gradient is heavily dependent on:
> 1. The derivative of the activation function: $\tanh'(z_j) = 1 - \tanh^2(z_j) \in [0, 1]$.
> 2. The largest eigenvalue (spectral radius) of the weight matrix $W_{hh}$.

---

### 3. Vanishing Gradients
When we train on long sequences, the distance $(T - k)$ is large.
- The derivative of $\tanh(z)$ is at most $1.0$ (when $z = 0$), but is usually much smaller (close to $0$) when activations saturate.
- If the eigenvalues of $W_{hh}$ are less than $1.0$, the product of $W_{hh}$ terms will shrink exponentially to $0$ as $(T - k)$ increases:
  $$\left\| \frac{\partial h_T}{\partial h_k} \right\| \to 0 \quad \text{as} \quad (T - k) \to \infty$$
- **Effect**: The gradient $\frac{\partial L_T}{\partial W_{hh}}$ for early steps becomes extremely close to zero. Consequently, weights associated with early inputs do not update, and the network fails to learn dependencies that span across many time steps.

---

### 4. Exploding Gradients
- If the largest eigenvalue of $W_{hh}$ is greater than $1.0$, and the hidden states do not fall into the saturation region of $\tanh$:
  $$\left\| \frac{\partial h_T}{\partial h_k} \right\| \to \infty \quad \text{as} \quad (T - k) \to \infty$$
- **Effect**: The gradients grow exponentially. During gradient descent updates, this causes massive weight updates, making training highly unstable. Weights can quickly overflow, resulting in `NaN` (Not a Number) values.

```
                  Gradient Behavior Over Time Steps (BPTT)
Early Steps (k = 1)                                           Final Step (T)
   |                                                                |
   |<==================== Backward Pass ============================|
   |                                                                |
   v (If product < 1)                                               v
[Vanishing: Gradient -> 0]                                    [Gradients Normal]
   |
   v (If product > 1)
[Exploding: Gradient -> Inf / NaN]
```

---

### 5. Solutions and Mitigations

To combat these issues, several techniques are used:

| Problem | Mitigation / Solution | How It Works |
| :--- | :--- | :--- |
| **Exploding Gradients** | **Gradient Clipping** | Rescales the gradient if its norm exceeds a predefined threshold (e.g., $g \leftarrow \text{threshold} \times \frac{g}{\|g\|}$). This prevents massive updates. |
| **Vanishing Gradients** | **Identity Initialization** | Initializing $W_{hh}$ close to the identity matrix ($I$) so that information is passed forward without shrinking. |
| **Vanishing Gradients** | **ReLU Activation** | Using ReLU instead of Tanh prevents the derivative from saturating to $0$ for positive values, though it requires care to prevent unbounded state growth. |
| **Both / Structural** | **LSTMs & GRUs** | Introduces **gates** and a **cell state** (constant error carousel) that allows gradients to flow backwards additively without exponential decay. |

> [!TIP]
> **Why LSTMs work**: The cell state in LSTMs acts as a "highway" where information can flow with minimal multiplication. The derivative of the cell state relation contains an additive term, which prevents the gradient from vanishing even over hundreds of time steps. This wraps up standard RNNs and leads directly to LSTM networks!
