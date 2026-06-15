# Lesson 10: Forward Propagation Mathematics

Welcome to my revision notes for **Lesson 10** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Step-by-Step Forward Pass Calculations**: Working out activation equations by hand.
2. **Vectorized Notation / Matrix Form**: Writing neural network equations for batch computing.

---

## 📝 Key Revision Points

### Mathematical Steps for a Single Neuron
For any neuron $j$ in layer $l$, the calculation consists of two steps:
1. **Weighted Sum ($z_j^{(l)}$)**:
   $$z_j^{(l)} = \sum_{k} w_{jk}^{(l)} a_k^{(l-1)} + b_j^{(l)}$$
2. **Activation Output ($a_j^{(l)}$)**:
   $$a_j^{(l)} = \sigma(z_j^{(l)})$$

### Matrix / Vectorized Form (For Efficient Computation)
To calculate forward propagation for an entire layer $l$ at once:
$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
$$\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$$

Where:
- $\mathbf{a}^{(l-1)}$ is the activation vector from the previous layer.
- $\mathbf{W}^{(l)}$ is the weight matrix for layer $l$.
- $\mathbf{b}^{(l)}$ is the bias vector for layer $l$.
- $\sigma$ is the element-wise activation function.
