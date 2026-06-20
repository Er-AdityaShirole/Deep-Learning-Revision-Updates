# Lesson 43: Padding and Strides in CNNs

Welcome to my revision notes for **Lesson 43** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **The Problems with Simple Convolutions**: Shrinking outputs and border information loss.
2. **Padding ($p$)**: Zero padding, Valid Padding, and Same Padding.
3. **Strides ($s$)**: Control step sizes.
4. **General Output Dimension Formula**.

---

## 📝 Key Revision Points

### The Problem: Why do we need Padding?
1. **Shrinking Output**: Every convolution reduces dimensions: $n \rightarrow n - f + 1$. In deep networks, the spatial dimensions quickly reduce to $1 \times 1$, limiting network depth.
2. **Loss of Border Info**: Pixels in corners are only visited once by the filter, whereas center pixels are visited multiple times. Border information is effectively ignored.

### Padding ($p$)
We pad the border of the input image with zeros.
- **Valid Padding**: No padding ($p = 0$). Output size is $n - f + 1$.
- **Same Padding**: Padding such that output size matches input size ($n \times n$).
  $$p = \frac{f - 1}{2}$$
  *(Since $p$ must be an integer, filter sizes $f$ are almost always odd, e.g., $3 \times 3$, $5 \times 5$)*.

### Strides ($s$)
Instead of moving the filter by 1 pixel, we can skip pixels. Stride $s$ defines the step size.

### General Dimension Formula:
For an input $n \times n$, filter $f \times f$, padding $p$, and stride $s$:

$$\text{Output Size} = \left\lfloor \frac{n + 2p - f}{s} \right\rfloor + 1$$

---

## 💻 Code Walkthrough (from `keras_padding_demo.ipynb`)

We build models to contrast **Valid** vs. **Same** padding in Keras:

### 1. Valid Padding API:
```python
model = Sequential()
# Input shape: (28, 28, 1) -> Output: (26, 26, 32)
model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(28,28,1)))
# Output: (24, 24, 32)
model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu'))
# Output: (22, 22, 32)
model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu'))
```

### 2. Same Padding with Stride 2 API:
```python
model = Sequential()
# Input shape: (28, 28, 1) -> Output: (14, 14, 32) (reduces by half due to stride=2)
model.add(Conv2D(32, kernel_size=(3,3), padding='same', strides=(2,2), activation='relu', input_shape=(28,28,1)))
# Output: (7, 7, 32)
model.add(Conv2D(32, kernel_size=(3,3), padding='same', strides=(2,2), activation='relu'))
```
