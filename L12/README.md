# Lesson 12: MNIST Digit Classification using ANN

Welcome to my revision notes for **Lesson 12** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **MNIST Dataset**: A collection of 70,000 grayscale 28x28 handwritten digits.
2. **Image Flattening**: Converting 2D spatial dimensions to 1D feature arrays.
3. **Keras Implementation**: Building a classification model with multiple outputs.
4. **Plotting Loss & Accuracy curves**: Checking for overfitting.

---

## 💻 Code walkthrough

This lesson uses the notebook [mnist_classification.ipynb](file:///C:/Users/lenovo/Desktop/Deep%20Learning/L12/mnist_classification.ipynb) for classifying handwritten digits.

### Step 1: Loading & Scaling Image Data
MNIST images are loadable natively from Keras. Pixels range from 0 to 255. We divide by 255 to scale them between 0 and 1:
```python
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train/255
X_test = X_test/255
```

### Step 2: Model Architecture
We use the `Flatten` layer to convert the 2D image matrices of shape `(28,28)` into a 1D vector of shape `(784,)`:
```python
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) # 10 classes (digits 0-9)
```

### Step 3: Compilation & Fitting
Since labels are integers rather than one-hot encoded vectors, we use **Sparse Categorical Crossentropy** loss:
```python
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=25, validation_split=0.2)
```

### Step 4: Analysis
Plotting training history curves helps diagnose model fit:
```python
# Plot Loss Curves
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')

# Plot Accuracy Curves
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
```
