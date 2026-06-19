# Lesson 38: Adam Optimizer

Welcome to my revision notes for **Lesson 38** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Adaptive Moment Estimation**: Combining momentum and RMSProp.
2. **First and Second Moments**: Tracking both gradient direction and gradient scale.
3. **Bias Correction**: Stabilizing early updates when moving averages start near zero.

---

## Visual Snapshot

![Adam optimizer introduction](Screenshot%202026-06-19%20153058.png)

Adam is one of the most commonly used optimizers because it combines the speed of momentum with the adaptive learning-rate behavior of RMSProp.

---

## Key Revision Points

### Why Adam is Powerful

Adam stands for **Adaptive Moment Estimation**.

It combines two important ideas:

| Idea | Optimizer Link | Role in Adam |
| :--- | :--- | :--- |
| Momentum | SGD with Momentum | Smooths the gradient direction |
| Adaptive scaling | RMSProp | Scales updates using squared gradients |

### First Moment: Momentum Term

Adam keeps an exponentially weighted moving average of gradients:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t$$

This behaves like momentum. It remembers the recent direction of gradients and reduces noisy zig-zag updates.

### Second Moment: RMSProp Term

Adam also keeps an exponentially weighted moving average of squared gradients:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2$$

This behaves like RMSProp. It adjusts the step size based on how large recent gradients have been.

### Bias Correction

At the beginning of training, $m_t$ and $v_t$ are initialized near zero. Adam corrects this early bias:

$$\hat{m_t} = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v_t} = \frac{v_t}{1 - \beta_2^t}$$

### Final Adam Update

The weight update is:

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v_t}} + \epsilon}\hat{m_t}$$

Common default values:

| Hyperparameter | Typical Value | Purpose |
| :---: | :---: | :--- |
| $\eta$ | `0.001` | Base learning rate |
| $\beta_1$ | `0.9` | Momentum decay |
| $\beta_2$ | `0.999` | Squared-gradient decay |
| $\epsilon$ | `1e-8` | Numerical stability |

### Optimizer Comparison

| Optimizer | What it remembers | Main strength |
| :--- | :--- | :--- |
| Momentum | Past gradients | Faster movement in consistent directions |
| AdaGrad | Sum of squared gradients | Per-parameter learning rates |
| RMSProp | Moving average of squared gradients | Fixes AdaGrad's shrinking-rate problem |
| Adam | Gradients + squared gradients | Combines momentum and adaptive scaling |

### Verdict

Adam is usually a strong starting optimizer for deep learning experiments. If Adam is not giving satisfactory results, RMSProp or careful hyperparameter tuning can be tried next.

### Main Takeaway

Adam combines momentum and RMSProp into one optimizer. It is fast, adaptive, stable in many cases, and widely used as a default choice for neural network training.

