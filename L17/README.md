# Lesson 17: Backpropagation - Part 3 (The Why - Intuition & Minima)

Welcome to my revision notes for **Lesson 17** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Why Backpropagation is Mathematically Necessary**: Understanding computational complexity.
2. **Concept of Minima**: Convex vs. Non-Convex loss surfaces.
3. **Weight Update Intuition**: How gradients guide optimization.

---

## 📝 Key Revision Points

### Why not Numerical Gradients?
Why do we calculate analytical gradients using the chain rule instead of numerical estimation?
- **Numerical Estimation**: Approximates derivatives by checking $\frac{f(x+\epsilon) - f(x)}{\epsilon}$. This requires evaluating the forward pass twice for *every single parameter*. In networks with millions of parameters, this is computationally impossible.
- **Backpropagation**: Calculates the exact analytical derivatives for *all parameters simultaneously* in a single backward pass, making training highly efficient.

### Weight Update Formula Intuition
$$W_{\text{new}} = W_{\text{old}} - \alpha \frac{\partial L}{\partial W}$$

- If the derivative $\frac{\partial L}{\partial W}$ is **positive** (slope is upward), subtracting it makes $W_{\text{new}}$ **smaller**, moving us leftward down the curve.
- If the derivative $\frac{\partial L}{\partial W}$ is **negative** (slope is downward), subtracting it makes $W_{\text{new}}$ **larger**, moving us rightward down the curve.

```
       Loss L
         |      
         |    \   / <- Positive slope (derivative > 0) -> Move left (W decreases)
         |     \_/  <- Local/Global Minima (slope = 0)
         +------------- W
```
