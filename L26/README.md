# Lesson 26: L1 and L2 Regularization

Welcome to my revision notes for **Lesson 26** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Ways to Reduce Overfitting**: Adding more data or reducing model complexity.
2. **Regularization**: Penalizing large weights to make the model simpler.
3. **L1 and L2 Regularization**: Adding weight penalties to the loss function.
4. **Weight Decay**: How L2 regularization shrinks weights during training.

---

## Key Revision Points

### Why Regularization is Needed
Overfitting often happens when a model becomes too complex and learns patterns that do not generalize.

Regularization adds a penalty to the loss function so the model is discouraged from using very large weights.

### L1 Regularization
L1 adds the absolute value of weights to the loss:

$$L' = L + \lambda \sum_i |w_i|$$

This can push some weights toward exactly zero, which creates sparsity.

### L2 Regularization
L2 adds the squared value of weights to the loss:

$$L' = L + \frac{\lambda}{2} \sum_i w_i^2$$

During gradient descent:

$$w_{new} = (1 - \eta\lambda)w_{old} - \eta \frac{\partial L}{\partial w}$$

The term $(1 - \eta\lambda)w_{old}$ shrinks the weight value. This is why L2 is also called **weight decay**.

### Main Takeaway
Regularization reduces overfitting by controlling model complexity through the loss function itself.
