# Lesson 33: Exponentially Weighted Moving Average

Welcome to my revision notes for **Lesson 33** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Moving Average**: Smoothing noisy values over time.
2. **Exponentially Weighted Average**: Giving more importance to recent observations.
3. **Role of Beta**: Controlling how smooth or reactive the average is.

---

## Key Revision Points

### Why Moving Average is Useful
Training signals such as gradients and losses can be noisy. A moving average helps smooth the noise so the trend becomes easier to follow.

### Exponentially Weighted Moving Average
The exponentially weighted moving average is calculated as:

$$v_t = \beta v_{t-1} + (1 - \beta)\theta_t$$

Here `v_t` is the current moving average, `v_{t-1}` is the previous moving average, `theta_t` is the current value, and `beta` controls smoothing.

### Effect of Beta
The value of `beta` decides how much past information is retained.

- Higher `beta`, such as `0.9`, gives a smoother curve.
- Lower `beta`, such as `0.5`, reacts faster to recent values.

Approximate memory of EWMA:

$$\frac{1}{1 - \beta}$$

So if `beta = 0.9`, the average roughly considers the last `10` values.

### Why This Matters in Optimizers
Momentum, RMSprop, and Adam all use the idea of exponentially weighted averages.

Instead of reacting only to the current gradient, they remember a smoothed version of past gradients or squared gradients.

### Main Takeaway
EWMA is the mathematical idea behind smoother optimizer updates. It balances memory of the past with response to the present.
