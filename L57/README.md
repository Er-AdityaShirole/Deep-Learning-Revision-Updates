# Lesson 57: RNN Sentiment Analysis in Keras

> Theme: convert text into numbers, feed it through an embedding layer, and let a SimpleRNN learn sequence context.

---

## Project Goal

This lesson moves from RNN theory into implementation. The main task is sentiment analysis using the IMDB movie review dataset.

```text
raw text -> integer tokens -> padded sequences -> embedding vectors -> SimpleRNN -> sentiment output
```

| Notebook | Purpose |
| :--- | :--- |
| `integer_encoding_simplernn.ipynb` | Integer encoding, padding, and a first SimpleRNN model |
| `sentiment_analysis_simplernn.ipynb` | Embedding layer plus SimpleRNN for IMDB sentiment classification |

---

## Integer Encoding

Neural networks cannot directly understand words. Text first needs to be converted into integer IDs.

```text
"go india" -> [9, 1]
"india india" -> [1, 1]
```

Keras `Tokenizer` builds a vocabulary from the training text:

```python
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)
sequences = tokenizer.texts_to_sequences(docs)
```

---

## Padding Sequences

Text examples usually have different lengths. RNN batches need equal-length sequences, so padding is used.

```python
from keras.utils import pad_sequences

sequences = pad_sequences(sequences, padding="post")
```

Post-padding adds zeros after shorter sequences:

```text
[9, 1] -> [9, 1, 0, 0, 0]
```

---

## Embedding Layer

Integer IDs are not meaningful by themselves. The embedding layer maps each word ID to a dense vector.

```python
model.add(Embedding(input_dim=10000, output_dim=2))
```

Unlike one-hot encoding, embeddings are compact and trainable.

---

## SimpleRNN Model

The IMDB sentiment model uses:

```python
model = Sequential()
model.add(Embedding(10000, 2))
model.add(SimpleRNN(32, return_sequences=False))
model.add(Dense(1, activation="sigmoid"))
```

The final sigmoid output predicts binary sentiment:

```text
0 -> negative review
1 -> positive review
```

The model is compiled with binary cross-entropy:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["acc"]
)
```

---

## Observations From Training

The embedding-based SimpleRNN improves validation accuracy compared with feeding raw integer sequences directly. Training accuracy can rise quickly, while validation accuracy may peak early and then fall, which signals overfitting.

---

## Revision Checkpoints

- Text must be tokenized before it can enter a neural network.
- `Tokenizer` creates a word-to-index mapping.
- `pad_sequences` creates fixed-length batches.
- `Embedding` converts integer IDs into dense trainable vectors.
- `SimpleRNN` reads the embedded sequence step by step.
- `return_sequences=False` returns only the final hidden state.
- Binary sentiment classification uses a sigmoid output and binary cross-entropy.

---

## Common Mistakes

| Mistake | Better thinking |
| :--- | :--- |
| Treating word IDs as real numeric quantities | Word IDs are labels; use embeddings for useful representations. |
| Forgetting padding | Batches need equal sequence lengths. |
| Using softmax for binary output | A single sigmoid unit is enough for binary sentiment. |
| Judging only training accuracy | Watch validation accuracy for overfitting. |

---

## One-Line Memory Hook

For text RNNs, token IDs are the address, embeddings are the meaning, and the RNN reads the meaning in order.
