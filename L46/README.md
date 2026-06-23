# Lesson 46: Comparing CNN vs. ANN (Computational Cost & Parameter counts)

Welcome to my revision notes for **Lesson 46** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Computational Comparison**: Direct comparison of parameter counts between ANNs (MLPs) and CNNs.
2. **Connectivity Patterns**: Fully Connected (ANN) vs. Sparsely Connected (CNN).
3. **Weight Sharing**: How CNNs recycle kernel weights across spatial locations.
4. **Pros and Cons**: Parameter efficiency and translation invariance.

---

## 📝 Key Revision Points

### 1. The Parameter Explosion in ANNs
For a color image of input size $W \times H \times C$ connected to a hidden layer of $H_1$ neurons:
- **ANN (Fully Connected)**:
  $$\text{Parameters} = (W \times H \times C) \times H_1 + H_1$$
- **CNN (Convolutional Layer)** with $K$ filters of size $f \times f \times C$:
  $$\text{Parameters} = (f \times f \times C + 1) \times K$$

#### Numerical Example:
Let's feed an image of size $226 \times 226 \times 3$ to:
- **An ANN layer** with $1,000$ hidden units:
  $$\text{Parameters} = (226 \times 226 \times 3) \times 1000 + 1000 = 153,229,000\text{ parameters!}$$
- **A CNN layer** with $32$ filters of size $3 \times 3 \times 3$:
  $$\text{Parameters} = (3 \times 3 \times 3 + 1) \times 32 = 896\text{ parameters!}$$

> [!IMPORTANT]
> The CNN layer uses **over 170,000 times fewer parameters** while retaining the spatial structure of the image! This makes deep networks computationally feasible.

### 2. Connectivity comparison: Sparse vs. Full
- **ANN (Dense Layer)**: Every input unit connects to every output unit (Global Connectivity).
- **CNN (Conv Layer)**: Each output pixel is computed from a local region (receptive field) of input pixels (Sparse Connectivity).

```
Dense Layer (ANN):     Conv Layer (CNN):
In1 ---\   /--- Out1    In1 ---\ 
In2 -----X----- Out2    In2 ----*---> Out1 (Local connection only)
In3 ---/   \--- Out3    In3 ---/
```
