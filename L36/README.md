# Lesson 36: AdaGrad Optimizer

Welcome to my revision notes for **Lesson 36** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Adaptive Gradients**: Changing the learning rate separately for each parameter.
2. **AdaGrad Update Rule**: Scaling updates using the history of squared gradients.
3. **Strengths and Weaknesses**: Why AdaGrad helps sparse features but slows down in long training.

---

## Visual Snapshot

![AdaGrad loss surface intuition](Screenshot%202026-06-19%20112618.png)

AdaGrad is useful when different parameters need different step sizes. On elongated loss surfaces, it can reduce aggressive movement in steep directions and allow better movement in flatter directions.

---

## Key Revision Points

### Why AdaGrad was Introduced

Normal gradient descent uses the same learning rate for every parameter:

$$w_t = w_{t-1} - \eta \nabla_w L$$

But in neural networks, every weight may not need the same update size. Some parameters receive frequent large gradients, while others receive rare or small gradients.

AdaGrad solves this by adapting the effective learning rate for each parameter.

### Core Idea

AdaGrad keeps a running sum of squared gradients:

$$G_t = G_{t-1} + g_t^2$$

Then it divides the current gradient by the square root of this accumulated value:

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$$

Here:

| Symbol | Meaning |
| :---: | :--- |
| $\eta$ | Base learning rate |
| $g_t$ | Current gradient |
| $G_t$ | Accumulated squared gradients |
| $\epsilon$ | Small value to avoid division by zero |

### Intuition

If a parameter has received large gradients many times, its accumulated value becomes large, so AdaGrad reduces its future update size.

If a parameter has received small or rare gradients, its accumulated value stays smaller, so it can still receive meaningful updates.

### Advantages

| Benefit | Explanation |
| :--- | :--- |
| Per-parameter learning rate | Each weight gets its own adaptive step size |
| Works well with sparse data | Rare features can still learn effectively |
| Less manual tuning | The optimizer adjusts learning rates automatically |

### Main Disadvantage

AdaGrad keeps adding squared gradients forever. Because of this, $G_t$ keeps increasing and the effective learning rate keeps decreasing:

$$\eta_{effective} = \frac{\eta}{\sqrt{G_t + \epsilon}}$$

After many epochs, updates can become extremely small and training may slow down too much.

### Main Takeaway

AdaGrad introduced the important idea of adaptive learning rates, but its continuously growing denominator can make it too slow for long neural network training.

