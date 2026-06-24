# Lesson 59: Backpropagation Through Time

> Theme: training an RNN means unfolding it across time, then applying backpropagation through the unfolded graph.

---

## Why Normal Backprop Needs An Upgrade

An RNN reuses the same weights at every time step. During forward propagation, the model creates a chain of hidden states:

```text
h1 -> h2 -> h3 -> ... -> ht
```

During training, the loss at later time steps depends on earlier hidden states. So gradients must flow backward through time.

This is called Backpropagation Through Time, or BPTT.

---

## Unrolling The RNN

The recurrent loop can be visualized as a deep network stretched across time:

```text
x1 -> [RNN] -> h1
x2 -> [RNN] -> h2
x3 -> [RNN] -> h3
```

The cells share the same weights, but the computational graph has multiple time-step copies for gradient calculation.

---

## Shared Weight Gradients

Because the same recurrent weights are used at every time step, the total gradient is the sum of contributions from all time steps.

```text
dL/dW = contribution from t1
      + contribution from t2
      + contribution from t3
      + ...
```

This is the key difference from a normal feedforward layer: one parameter matrix affects many positions in the sequence.

---

## Chain Rule Across Time

For a basic RNN:

```text
h_t = f(W_xh x_t + W_hh h_(t-1) + b_h)
```

The loss gradient at time `t` can flow backward to output weights, the current hidden state, recurrent weights, earlier hidden states, and input-to-hidden weights.

---

## Vanishing And Exploding Gradients

BPTT repeatedly multiplies gradients through time. If the repeated factors are small, gradients vanish. If they are large, gradients explode.

| Problem | Effect |
| :--- | :--- |
| Vanishing gradient | Earlier time steps learn very slowly |
| Exploding gradient | Updates become unstable and loss may blow up |

This is why vanilla RNNs struggle with long-term dependencies.

---

## Practical Fixes

| Issue | Common fix |
| :--- | :--- |
| Exploding gradients | Gradient clipping |
| Long-term memory loss | LSTM or GRU |
| Very long sequences | Truncated BPTT |
| Training instability | Careful initialization and learning rate tuning |

Truncated BPTT limits how far back gradients are propagated, which reduces computation and instability.

---

## Revision Checkpoints

- BPTT is backpropagation applied to an RNN unrolled across time.
- Recurrent weights are shared, so gradients are accumulated across time steps.
- Later losses can affect earlier hidden states through the chain rule.
- Repeated gradient multiplication causes vanishing and exploding gradients.
- Gradient clipping helps exploding gradients.
- LSTM and GRU improve long-term dependency learning.

---

## Common Mistakes

| Mistake | Better thinking |
| :--- | :--- |
| Thinking every unrolled cell has separate weights | They are copies of the same shared parameters. |
| Ignoring earlier time-step gradients | Earlier inputs influence later hidden states. |
| Assuming BPTT fully solves long memory | Vanilla RNNs still suffer from vanishing gradients. |
| Forgetting gradient accumulation | Shared weights collect gradient from all time positions. |

---

## One-Line Memory Hook

BPTT trains an RNN by unfolding time, sending error backward, and adding up the blame for shared weights.
