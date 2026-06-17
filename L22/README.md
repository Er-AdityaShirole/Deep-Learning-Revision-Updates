# Lesson 22: Early Stopping

Welcome to my revision notes for **Lesson 22** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Overfitting During Training**: Training loss keeps decreasing while validation loss starts increasing.
2. **Early Stopping Idea**: Stop training when validation performance stops improving.
3. **Keras Callback**: Using `EarlyStopping` with `monitor`, `patience`, and `restore_best_weights`.

---

## Key Revision Points

### Why Early Stopping is Needed
If a model is trained for too many epochs, it can start memorizing the training data. A common pattern is:

```text
Training loss   -> keeps decreasing
Validation loss -> decreases first, then starts increasing
```

That means the model is no longer generalizing well.

### Main Idea
Early stopping watches a metric such as `val_loss`. If the metric does not improve for a fixed number of epochs, training is stopped.

### Important Arguments
- `monitor`: Metric to track, usually `val_loss`.
- `patience`: Number of epochs to wait before stopping.
- `min_delta`: Minimum improvement required to count as progress.
- `mode`: Whether the metric should be minimized or maximized.
- `restore_best_weights`: Restores weights from the best epoch instead of the last epoch.

### Keras Example

```python
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=3500,
    callbacks=[callback]
)
```

Early stopping is useful because it reduces overfitting and saves unnecessary training time.
