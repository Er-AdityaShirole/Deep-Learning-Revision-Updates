# Lesson 49: Cat vs. Dog Image Classification Project

Welcome to my revision notes for **Lesson 49** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Kaggle Dataset API Integration**: Downloading the Dogs vs. Cats dataset.
2. **Data Pipeline Loading**: Loading images efficiently from directory paths.
3. **Normalizing Images**: Scaling pixel values using TF datasets map API.
4. **ANN vs. CNN Performance**: Testing architectures and diagnosing overfitting.

---

## 💻 Code Walkthrough (from `main.ipynb`)

In this project, we download the Kaggle dataset containing 20,000 images of cats and dogs, and build a convolutional model.

### Step 1: Directory Data Loading
We load images from folders into memory-efficient `tf.data.Dataset` generators:
```python
import tensorflow as tf
from tensorflow import keras

train_df = keras.utils.image_dataset_from_directory(
    directory="C:/Users/lenovo/Desktop/Deep Learning/L49/train",
    labels="inferred",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_df = keras.utils.image_dataset_from_directory(
    directory="C:/Users/lenovo/Desktop/Deep Learning/L49/train",
    labels="inferred",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=42
)
```

### Step 2: Normalization Mapping
Images are loaded as integers $[0, 255]$. We normalize them to float $[0, 1]$:
```python
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_df = train_df.map(normalize)
val_df = val_df.map(normalize)
```

### Step 3: Model Architecture
We stack Conv2D, BatchNormalization, and MaxPooling layers:
```python
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid')) # Binary classification (cat vs dog)
```

### Step 4: Overfitting diagnosis
When training, the training accuracy rises to $95\%+$ but validation accuracy plateaus around $78\%$, with validation loss increasing. This indicates severe **overfitting** due to high parameters and small data variance, which leads us directly to **Data Augmentation** in the next lesson.
