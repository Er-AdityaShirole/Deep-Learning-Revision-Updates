# Lesson 50: Data Augmentation in CNNs

Welcome to my revision notes for **Lesson 50** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **What is Data Augmentation?**: Artificial dataset expansion.
2. **Overfitting Mitigation**: Using random transformations to regularize model learning.
3. **Keras ImageDataGenerator API**: Executing rotations, shifting, shearing, zooming, and flipping.

---

## 📝 Key Revision Points

### Why Data Augmentation?
When training deep neural networks on small datasets, models easily memorize the exact training images, causing overfitting. Data Augmentation applies random transformations to the images, creating slightly modified variations.
- The model never sees the exact same image twice, forcing it to learn generic patterns (e.g. ears, snout) rather than specific pixel locations.

### Common Image Transformations
- **Rotation**: Rotate the image by a random degree (e.g., $40^\circ$).
- **Width/Height Shift**: Shift the image horizontally or vertically (e.g. by $20\%$).
- **Shearing**: Skew/tilt the image along an axis.
- **Zooming**: Randomly zoom in or out.
- **Horizontal Flip**: Flip the image horizontally (dogs/cats are still dogs/cats when flipped horizontally; however, vertical flipping is usually avoided as animals don't stand upside down).
- **Fill Mode**: What to do with empty pixels created after shifting (e.g. `nearest` fills them with neighboring pixels).

```
Original Image ---> [Rotation + Horizontal Flip + Zoom] ---> Augmented Variations
```

---

## 💻 Code Example (from screenshots)

Setting up the data generator:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Apply during training in model.fit()
# This yields augmented images on-the-fly without saving them to disk, conserving storage.
```
