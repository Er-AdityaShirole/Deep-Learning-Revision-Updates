# Lesson 8: Multi-layer Perceptron (MLP) Notation

Welcome to my revision notes for **Lesson 8** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Neural Network Architecture Structure**: Inputs, Hidden layers, Outputs.
2. **Index Notation**: Standard mathematical notation for weights, biases, and activations across layers.
3. **Vectorized Notation**: Preparing equations for matrix operations.

---

## 📝 Key Revision Points

### Standard Notation Rules

To represent a neural network mathematically, we use index superscripts and subscripts:

$$w_{jk}^{(l)}$$
- $l$: The **target layer** of the weight (where the signal is going).
- $j$: The index of the **neuron in the target layer** $l$.
- $k$: The index of the **neuron in the source layer** $l-1$.

For example, $w_{32}^{(2)}$ represents the weight from neuron $2$ in the $1$st layer (hidden layer) to neuron $3$ in the $2$nd layer (output/next layer).

$$b_j^{(l)}$$
- The bias of the $j$-th neuron in the $l$-th layer.

$$a_j^{(l)}$$
- The activation (output output) of the $j$-th neuron in the $l$-th layer. (For the input layer $l=0$, $a_k^{(0)} = x_k$).

```
                Layer (l-1)       Layer (l)
                Neuron k -------- w_{jk}^{(l)} --------> Neuron j (bias b_j^{(l)})
                                                         Activation: a_j^{(l)}
```
