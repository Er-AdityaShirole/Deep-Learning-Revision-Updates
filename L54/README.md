# Lesson 54: Keras Functional API and Non-Linear Model Graphs

> Theme: move beyond the straight line of `Sequential` and build neural networks as flexible computation graphs.

---

## Why Functional API Exists

`Sequential` is perfect when the model is a simple stack:

```text
input -> layer -> layer -> layer -> output
```

But many real models are not a single straight pipe. They may have multiple inputs, multiple outputs, shared layers, skip connections, branching paths, merged features, or different heads for different tasks.

The Keras Functional API lets us describe these models by explicitly connecting tensors.

---

## The Mental Model

In Functional API, each layer is a function call on a tensor.

```python
from keras.layers import Input, Dense
from keras.models import Model

x = Input(shape=(3,))
h1 = Dense(128, activation="relu")(x)
h2 = Dense(64, activation="relu")(h1)
y = Dense(1, activation="linear")(h2)

model = Model(inputs=x, outputs=y)
```

Read this as:

```text
x enters h1
h1 feeds h2
h2 produces y
Model binds x as the input and y as the output
```

The graph is created by the tensor connections, not by adding layers one by one.

---

## Functional API vs Sequential

| Feature | Sequential | Functional API |
| :--- | :--- | :--- |
| Simple linear stack | Excellent | Excellent |
| Multiple inputs | Not suitable | Natural |
| Multiple outputs | Limited | Natural |
| Skip connections | Not suitable | Natural |
| Shared layers | Not suitable | Natural |
| Model visualization | Basic | Very clear graph |

Sequential is a neat notebook. Functional API is a wiring board.

---

## Multi-Output Neural Networks

A multi-output network has one shared body and more than one prediction head.

```python
from keras.layers import Input, Dense
from keras.models import Model

inputs = Input(shape=(3,))

h = Dense(128, activation="relu")(inputs)
h = Dense(64, activation="relu")(h)

price_output = Dense(1, activation="linear", name="price")(h)
category_output = Dense(1, activation="sigmoid", name="category")(h)

model = Model(
    inputs=inputs,
    outputs=[price_output, category_output]
)
```

This is useful when one input example must produce different kinds of answers, such as predicting price and category, classifying an object and estimating a bounding box, or predicting age and gender.

---

## How Loss Works With Multiple Outputs

Each output has its own loss.

```text
L_total = L_1 + L_2
```

Sometimes we weight the losses:

```text
L_total = alpha * L_1 + beta * L_2
```

During backpropagation, shared layers receive the combined gradient:

```text
dL_total/dw = dL_1/dw + dL_2/dw
```

If both tasks agree, the update is strong. If they disagree, their gradients partially cancel. The optimizer follows the net effect.

---

## Compiling Multi-Output Models

Different heads may need different losses.

```python
model.compile(
    optimizer="adam",
    loss={
        "price": "mse",
        "category": "binary_crossentropy"
    },
    loss_weights={
        "price": 0.7,
        "category": 0.3
    },
    metrics={
        "price": ["mae"],
        "category": ["accuracy"]
    }
)
```

Loss weights are not decoration. They decide how loudly each task speaks during training.

---

## Revision Checkpoints

- Functional API represents models as directed graphs of tensors.
- `Input()` creates the model input tensor.
- Calling a layer on a tensor connects that layer into the graph.
- `Model(inputs=..., outputs=...)` finalizes the graph.
- It is the right tool for branching, merging, multiple outputs, and non-linear architectures.
- Multi-output models combine losses into one total loss.
- Shared layers learn features that help all connected tasks.

---

## Common Mistakes

| Mistake | Fix |
| :--- | :--- |
| Forgetting to call a layer on the previous tensor | Use `h2 = Dense(...)(h1)`, not only `Dense(...)`. |
| Using one loss for incompatible outputs | Use separate losses per output head. |
| Letting one loss dominate | Use `loss_weights` to balance task influence. |
| Treating output heads as independent models | Remember the shared trunk receives combined gradients. |

---

## One-Line Memory Hook

Sequential stacks layers; Functional API connects ideas.
