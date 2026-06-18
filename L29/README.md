# Lesson 29: Weight Initialization Techniques

Welcome to my revision notes for **Lesson 29** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Why Initialization Matters**: Initial weights decide how information and gradients flow at the start of training.
2. **Zero Initialization Problem**: Same weights make neurons learn the same features.
3. **Random Initialization**: Breaking symmetry so hidden neurons can specialize.

---

## Key Revision Points

### Why We Need Weight Initialization
Before training starts, a neural network needs initial values for weights and biases. These values should be chosen carefully because bad initialization can cause slow training, unstable activations, and poor convergence.

The goal is to start with weights that are neither too large nor too small.

### Zero Initialization
If all weights are initialized with zero, every neuron in the same layer receives the same input and produces the same output.

During backpropagation, they also receive similar gradients, so they keep updating in the same way. This is called the **symmetry problem**.

$$w_1 = w_2 = w_3 = 0$$

All neurons behave like copies of each other, so adding more neurons does not increase learning power.

### Same Constant Initialization
Initializing every weight with the same non-zero value, such as `0.5`, also creates the same symmetry issue.

The model may start with non-zero activations, but neurons in the same layer still learn identical features.

### Random Initialization
Random initialization gives different starting weights to different neurons:

$$w \sim random\ distribution$$

This breaks symmetry and allows each neuron to learn a different feature.

### Main Takeaway
Never initialize all hidden-layer weights with the same value. Random initialization is needed to break symmetry, but the random values must also be scaled properly, which leads to Xavier and He initialization.
