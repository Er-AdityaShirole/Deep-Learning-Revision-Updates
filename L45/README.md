# Lesson 45: Classic CNN Architectures - LeNet-5

Welcome to my revision notes for **Lesson 45** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **LeNet-5 (Yann LeCun, 1998)**: The network that launched CNNs.
2. **LeNet-5 Layer Sequence**: Convolutions, Average Pooling, and Dense Layers.
3. **Keras Implementation**: Replicating LeNet-5.

---

## 📝 Key Revision Points

### LeNet-5 Architecture Sequence
LeNet-5 was designed to recognize handwritten digits on check slips (MNIST). It has the following sequence:

```
Input (32x32x1) -> C1: Conv (5x5, 6 filters) -> S2: AvgPool (2x2, stride 2) -> C3: Conv (5x5, 16 filters) -> S4: AvgPool (2x2, stride 2) -> F5: Flatten/Conv (120 units) -> F6: Dense (84 units) -> Output: Softmax (10 units)
```

1. **Input**: $32 \times 32 \times 1$ grayscale image.
2. **C1 (Convolution)**: 6 filters of size $5 \times 5$. Output: $28 \times 28 \times 6$. Activation: `tanh`.
3. **S2 (Average Pooling)**: Window $2 \times 2$, stride 2. Output: $14 \times 14 \times 6$.
4. **C3 (Convolution)**: 16 filters of size $5 \times 5$. Output: $10 \times 10 \times 16$. Activation: `tanh`.
5. **S4 (Average Pooling)**: Window $2 \times 2$, stride 2. Output: $5 \times 5 \times 16$.
6. **Flatten / C5**: Flattened to $400$ inputs, connected to a Dense layer of $120$ units with `tanh`.
7. **F6 (Dense)**: $84$ units with `tanh`.
8. **Output**: 10 units with `softmax` (for classification of digits 0-9).

---

## 💻 Code Walkthrough (from `lenet-demo.ipynb`)

Replicating LeNet-5 in Keras:

```python
from keras import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

model = Sequential()

# Layer 1: Conv (5x5, 6 filters)
model.add(Conv2D(6, kernel_size=(5, 5), padding='valid', activation='tanh', input_shape=(32, 32, 1)))
# Layer 2: Average Pool (2x2, stride 2)
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# Layer 3: Conv (5x5, 16 filters)
model.add(Conv2D(16, kernel_size=(5, 5), padding='valid', activation='tanh'))
# Layer 4: Average Pool (2x2, stride 2)
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# Flatten and fully connected layers
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(10, activation='softmax'))

# Output summary matches LeNet-5 exactly
model.summary()
```
