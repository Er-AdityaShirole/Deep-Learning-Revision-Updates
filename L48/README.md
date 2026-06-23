# Lesson 48: Backpropagation in CNN: Part 2 (Conv, MaxPool, & Flatten Layers)

Welcome to my revision notes for **Lesson 48** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Backpropagating through Convolution**: Computing gradients of the loss with respect to input activations.
2. **Backpropagating through MaxPooling**: Re-routing error gradients only to the maximum activation index.
3. **Weight Sharing Updates**: Accumulating gradients for shared filter parameters.

---

## 📝 Key Revision Points

### 1. Backpropagation through Max Pooling
MaxPooling layers have no learnable parameters, so we do not compute weight gradients. However, we must pass the incoming gradient $\frac{\partial L}{\partial Y}$ backward to the input of the pooling layer $\frac{\partial L}{\partial X}$.

> [!IMPORTANT]
> **Gradient Routing**: During the forward pass of Max Pooling, the index of the maximum value in each window is recorded (cached). During the backward pass, the gradient is routed **exclusively to the position of that maximum value**. All other non-maximum positions receive a gradient of $0$.

$$\frac{\partial L}{\partial X_{i, j}} = \begin{cases} \frac{\partial L}{\partial Y_{r, c}} & \text{if } X_{i, j} \text{ was the maximum in window }(r, c) \\ 0 & \text{otherwise} \end{cases}$$

```
Forward Pass:              Backward Pass: (Gradient = dL/dY)
[ [ 2,  8 ],                [ [ 0,  dL/dY ],
  [ 1,  5 ] ] -> Max is 8     [ 0,    0   ] ]
```

### 2. Backpropagation through Convolutional Layers
To calculate the gradient of the loss with respect to the input of the convolutional layer ($X$) so that the error can be passed to previous layers, we convolve the incoming gradient $\frac{\partial L}{\partial Z}$ with a **flipped version of the filter** $W$:

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} * \text{rotate180}(W)$$
This operation behaves like a deconvolution/transpose convolution step during the backward pass.
