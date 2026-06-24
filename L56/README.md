# Lesson 56: RNN Forward Propagation and Architecture

> Theme: an RNN reads a sequence step by step and keeps a small memory of what it has already seen.

---

## Why RNN Architecture Exists

A normal dense network treats the full input as one fixed vector. That works for tabular features, but sequence problems need a model that can process ordered inputs:

```text
x1, x2, x3, ..., xt
```

An RNN solves this by applying the same cell repeatedly across time. At each time step, the cell receives the current input `x_t` and the previous hidden state `h_(t-1)`.

---

## Core Forward Propagation Equation

The basic RNN update is:

```text
h_t = f(W_xh x_t + W_hh h_(t-1) + b_h)
```

For a final output:

```text
y_t = g(W_hy h_t + b_y)
```

| Symbol | Meaning |
| :--- | :--- |
| `x_t` | Input at time step `t` |
| `h_t` | Hidden state after reading `x_t` |
| `h_(t-1)` | Memory from the previous time step |
| `W_xh` | Input-to-hidden weight matrix |
| `W_hh` | Hidden-to-hidden recurrent weight matrix |
| `W_hy` | Hidden-to-output weight matrix |

---

## Shared Weights Across Time

The same RNN cell is reused at every time step.

```text
x1 -> [RNN cell] -> h1
x2 -> [same cell] -> h2
x3 -> [same cell] -> h3
```

This weight sharing keeps the parameter count independent of sequence length and lets the same transition logic be learned for every position.

---

## Hidden State as Memory

The hidden state summarizes previous inputs:

```text
h_t = memory of x1 through xt
```

For text, this means remembering earlier words. For time series, it means carrying previous trends. For speech, it means preserving information from earlier sound frames.

---

## Forward Pass Flow

For a sequence of three inputs:

```text
h0 = initial hidden state
h1 = f(W_xh x1 + W_hh h0 + b_h)
h2 = f(W_xh x2 + W_hh h1 + b_h)
h3 = f(W_xh x3 + W_hh h2 + b_h)
y  = g(W_hy h3 + b_y)
```

In many-to-one problems, such as sentiment analysis, the final hidden state is commonly used for prediction.

---

## Important Shapes

| Quantity | Typical shape idea |
| :--- | :--- |
| Input sequence | `(time_steps, input_features)` |
| Hidden state | `(hidden_units,)` |
| Recurrent weights | `(hidden_units, hidden_units)` |
| Input weights | `(input_features, hidden_units)` |

The number of hidden units controls the size of the RNN's memory.

---

## Revision Checkpoints

- RNNs process sequential data one time step at a time.
- The hidden state carries information forward.
- The same weights are reused at every time step.
- Forward propagation combines current input and previous hidden state.
- A final prediction can be made from the last hidden state.

---

## Common Mistakes

| Mistake | Better thinking |
| :--- | :--- |
| Thinking each time step has separate weights | The RNN cell shares weights across all time steps. |
| Treating hidden state as raw history | It is a learned compressed summary, not a full copy. |
| Ignoring the initial hidden state | `h0` starts the recurrence and is usually zeros. |
| Assuming RNNs remember everything | Vanilla RNNs struggle with long-term dependencies. |

---

## One-Line Memory Hook

An RNN is a looped neural network that updates its memory every time it reads the next item.
