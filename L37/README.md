# Lesson 37: RMSProp Optimizer

Welcome to my revision notes for **Lesson 37** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **RMSProp Motivation**: Fixing AdaGrad's continuously shrinking learning rate.
2. **Squared Gradient Moving Average**: Using EWMA instead of a full historical sum.
3. **Comparison with AdaGrad and Adam**: Understanding where RMSProp fits in optimizer evolution.

---

## Visual Snapshot

![RMSProp motivation](Screenshot%202026-06-19%20151325.png)

RMSProp stands for **Root Mean Square Propagation**. It improves AdaGrad by using a moving average of squared gradients instead of accumulating every squared gradient forever.

---

## Key Revision Points

### Problem with AdaGrad

AdaGrad's denominator keeps growing because it adds all past squared gradients:

$$G_t = G_{t-1} + g_t^2$$

As $G_t$ becomes very large, the effective learning rate becomes very small. This can make learning almost stop during long training.

### RMSProp Core Idea

RMSProp replaces the full sum with an exponentially weighted moving average of squared gradients:

$$S_t = \beta S_{t-1} + (1 - \beta)g_t^2$$

Then the parameter update becomes:

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{S_t + \epsilon}} g_t$$

Here:

| Symbol | Meaning |
| :---: | :--- |
| $S_t$ | Moving average of squared gradients |
| $\beta$ | Decay rate for past squared gradients |
| $\eta$ | Base learning rate |
| $\epsilon$ | Numerical stability constant |

### Why RMSProp Helps

RMSProp remembers recent squared gradients more strongly and slowly forgets older ones.

This prevents the denominator from growing without limit, so the optimizer can keep learning even after many epochs.

### AdaGrad vs RMSProp

| Feature | AdaGrad | RMSProp |
| :--- | :--- | :--- |
| Gradient history | Full accumulated sum | Exponentially weighted average |
| Learning rate trend | Keeps shrinking | Remains more flexible |
| Best use case | Sparse features | Deep networks and non-stationary objectives |
| Main issue | Can become too slow | Needs tuning of $\eta$ and $\beta$ |

### Limitation

RMSProp improves AdaGrad, but it still only adapts learning rates using squared gradients. It does not directly use momentum from the first moment of the gradient.

This is exactly where Adam comes in.

### Main Takeaway

RMSProp fixes AdaGrad's major weakness by replacing the ever-growing squared-gradient sum with a moving average. It keeps adaptive learning rates practical for deep neural networks.

