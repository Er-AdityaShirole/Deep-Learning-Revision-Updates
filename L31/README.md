# Lesson 31: Batch Normalization

Welcome to my revision notes for **Lesson 31** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Batch Normalization**: Normalizing layer inputs during training.
2. **Internal Covariate Shift**: Distribution changes inside hidden layers.
3. **BatchNorm in Keras**: Adding `BatchNormalization` layers to a neural network.

---

## Key Revision Points

### What is Batch Normalization?
Batch normalization normalizes the inputs of a layer using the mean and variance of the current mini-batch.

$$\mu_B = \frac{1}{m}\sum x_i$$

$$\sigma_B^2 = \frac{1}{m}\sum (x_i - \mu_B)^2$$

Then each value is normalized:

$$\hat{x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

### Learnable Scale and Shift
After normalization, BatchNorm applies two learnable parameters:

$$y_i = \gamma \hat{x_i} + \beta$$

Here `gamma` controls scale, `beta` controls shift, and `epsilon` avoids division by zero.

### Internal Covariate Shift
During training, earlier layers keep changing their weights. Because of this, the input distribution seen by later layers also changes.

Batch normalization reduces this instability by keeping intermediate activations more controlled.

### Benefits

- Faster and more stable training.
- Allows higher learning rates in many cases.
- Reduces sensitivity to weight initialization.
- Can provide a small regularization effect.

### Keras Usage
In Keras, BatchNorm is commonly added after a dense or convolution layer and before/after activation depending on the architecture choice.

```python
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))
```

### Main Takeaway
Batch normalization stabilizes hidden-layer distributions, making deep networks easier and faster to train.
