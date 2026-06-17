# Lesson 24: Dropout Layers

Welcome to my revision notes for **Lesson 24** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Dropout as Regularization**: Reducing overfitting by randomly disabling neurons during training.
2. **How Dropout Works**: Different neurons are dropped in different epochs.
3. **Why it Works**: The network cannot depend too much on a fixed set of neurons.

---

## Key Revision Points

### Problem: Overfitting
Overfitting happens when the model learns the training data too closely and performs poorly on unseen data.

Dropout is a regularization technique designed to reduce this dependency.

### Main Idea
During training, dropout randomly turns off a fraction of neurons:

```text
Epoch 1 -> some neurons are dropped
Epoch 2 -> a different set is dropped
Epoch 3 -> another different set is dropped
```

If dropout probability is `p = 0.5`, then roughly half the selected layer's neurons are ignored during one training pass.

### Why Dropout Helps
- It prevents co-adaptation between neurons.
- It makes the network more robust.
- It behaves like training many smaller subnetworks.
- It improves generalization on test data.

### Important Note
Dropout is active only during training. During prediction/testing, all neurons are used.
