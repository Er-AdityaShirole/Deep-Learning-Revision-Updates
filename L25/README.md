# Lesson 25: Dropout Code Example

Welcome to my revision notes for **Lesson 25** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Dropout in Regression and Classification**: Applying dropout in practical neural network code.
2. **Train/Test Comparison**: Observing how dropout affects generalization.
3. **Dropout Rate Effect**: Understanding the value of `p` and its impact on training.
4. **Limitations**: Dropout can delay convergence and change the loss curve behavior.

---

## Key Revision Points

### Dropout in Keras
Dropout is added between dense layers:

```python
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=2))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

### Regression and Classification Use
The lecture connects dropout with both regression and classification examples. The same core idea applies:

```text
Train with random neuron dropping -> test with full network
```

This reduces overfitting and helps the model avoid memorizing training data.

### Effect of Dropout Probability
- Small dropout: mild regularization.
- Large dropout: stronger regularization, but can make learning slower.
- Too much dropout: model may underfit.

### Drawbacks
- Training can take longer to converge.
- The loss curve may fluctuate because the network changes during training.
- Choosing the right dropout rate requires experimentation.

Dropout is powerful, but it should be tuned carefully.
