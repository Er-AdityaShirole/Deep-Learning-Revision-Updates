# Lesson 44: Pooling Layers in CNNs

Welcome to my revision notes for **Lesson 44** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Why Pooling?**: The challenge of Translation Invariance.
2. **Max Pooling vs. Average Pooling**: Mechanics and differences.
3. **Advantages and Disadvantages of Pooling**.

---

## 📝 Key Revision Points

### The Problem: Translation Invariance
If we train a classifier to detect a feature (like an eye), and the eye moves 1 or 2 pixels in the image, the raw activations in the feature map shift completely. We need the model to be **robust to small translation/shifting of features**.

### What is Pooling?
Pooling downsamples the feature map by taking a summary value over a window (e.g., $2 \times 2$ with stride 2).

- **Max Pooling**: Extracts the **maximum** value from the window. It keeps the most prominent feature activation.
- **Average Pooling**: Computes the **average** value of the window. It smooths the feature activations.

```
Max Pooling on [ [10, 20], [5, 8] ] ===> Output: 20
Avg Pooling on [ [10, 20], [5, 8] ] ===> Output: 10.75
```

### Properties of Pooling Layers
- **No Learnable Parameters**: Pooling layers have $W = 0$ and $b = 0$. They only have fixed hyperparameters (pool size $f$, stride $s$).
- **Shrinks Dimensions**: Reduces spatial dimensions to speed up downstream computations.
- **Translation Invariance**: Since max pooling extracts the max value in a local neighborhood, if a feature moves slightly but remains in that neighborhood, the output value is unchanged.

### Disadvantages
- **Loss of Information**: Downsampling throws away detailed spatial representations, which can degrade performance in tasks like image segmentation or object detection.

---

## 💻 Code Walkthrough (from `keras_pooling_demo.ipynb`)

Stitching Conv2D with MaxPooling2D layers:

```python
model = Sequential()
# Input: (28, 28, 1) -> Output: (26, 26, 32)
model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(28,28,1)))
# Pool 2x2 with stride 2 -> Output: (13, 13, 32)
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
# Output: (11, 11, 32)
model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu'))
# Pool 2x2 with stride 2 -> Output: (5, 5, 32)
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
```
