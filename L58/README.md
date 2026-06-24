# Lesson 58: Types of RNN Architectures

> Theme: RNNs are flexible because the input and output can both be sequences or single values.

---

## Why RNN Types Matter

Different sequence problems need different input-output patterns. Sometimes one sequence produces one label. Sometimes one input produces a full sequence. Sometimes each time step needs its own output.

RNN architecture is chosen based on the shape of the problem.

---

## Main RNN Patterns

| Type | Input | Output | Example |
| :--- | :--- | :--- | :--- |
| One-to-one | Single input | Single output | Standard feedforward classification |
| One-to-many | Single input | Sequence output | Image captioning |
| Many-to-one | Sequence input | Single output | Sentiment analysis |
| Many-to-many, same length | Sequence input | Sequence output | Part-of-speech tagging |
| Many-to-many, different length | Sequence input | Sequence output | Machine translation |

---

## One-to-Many

One input produces a sequence.

```text
x -> y1, y2, y3, ..., yt
```

Example:

```text
image -> caption words
```

The model receives one context vector and generates outputs step by step.

---

## Many-to-One

A full sequence produces one final output.

```text
x1, x2, x3, ..., xt -> y
```

Examples include movie review sentiment, spam detection, disease prediction from patient time-series, and stock movement classification.

The final hidden state usually acts as the sequence summary.

---

## Many-to-Many With Same Length

Each input time step has a matching output time step.

```text
x1 -> y1
x2 -> y2
x3 -> y3
```

Examples include named entity recognition, part-of-speech tagging, and frame-level activity recognition.

The model must return an output for every time step, so `return_sequences=True` is often needed in Keras.

---

## Many-to-Many With Different Length

The input and output are both sequences, but their lengths may differ.

```text
x1, x2, x3 -> y1, y2, y3, y4, y5
```

Examples include machine translation, summarization, and chatbot response generation.

This often uses an encoder-decoder structure:

```text
input sequence -> encoder -> context -> decoder -> output sequence
```

---

## Revision Checkpoints

- RNNs describe a family of sequence patterns.
- Many-to-one is common for classification over text or time-series.
- One-to-many generates a sequence from one input.
- Many-to-many can produce aligned or unaligned sequence outputs.
- Encoder-decoder models handle different input and output lengths.
- `return_sequences=True` is needed when later layers require every hidden state.

---

## Common Mistakes

| Mistake | Better thinking |
| :--- | :--- |
| Using many-to-one for token-level labeling | Token-level tasks need an output per time step. |
| Assuming input and output lengths are always equal | Translation and summarization often have different lengths. |
| Forgetting `return_sequences=True` | Required when passing the full sequence onward. |
| Treating one-to-one as an RNN-only case | One-to-one is the standard feedforward setup. |

---

## One-Line Memory Hook

RNN type is decided by whether the problem starts with one thing or many, and ends with one thing or many.
