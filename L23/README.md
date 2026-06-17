# Lesson 23: Feature Scaling in Neural Networks

Welcome to my revision notes for **Lesson 23** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Why Feature Scaling Matters**: Neural networks train better when input features are on similar scales.
2. **Normalization / Standardization**: Transforming input values before passing them into the model.
3. **Effect on Training**: Faster convergence and better validation accuracy after scaling.

---

## Key Revision Points

### Why Scaling is Important
Neural networks use gradient-based optimization. If one feature has very large values and another has small values, the loss surface can become uneven and training may become slow or unstable.

Feature scaling makes optimization easier by putting features into comparable ranges.

### Common Scaling Methods

#### Standardization
Centers values around mean `0` with standard deviation `1`:

$$x' = \frac{x - \mu}{\sigma}$$

#### Normalization
Scales values into a fixed range such as `0` to `1`:

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### Practical Impact
After scaling:
- Gradients behave more consistently.
- Training can converge faster.
- Validation accuracy can improve.
- The optimizer does not get dominated by one large-scale feature.

Input normalization is one of the simplest improvements before training an ANN.
