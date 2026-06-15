# Lesson 13: Graduate Admission Prediction using ANN

Welcome to my revision notes for **Lesson 13** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Graduate Admission Prediction**: Predicting the probability of admission (ranging from 0 to 1).
2. **ANN Regression**: Designing neural networks for continuous value outputs.
3. **Evaluating Regression**: Utilizing $R^2$ score to measure performance.

---

## 💻 Code Walkthrough

Based on the Kaggle notebook and dataset used in this lesson:

### Step 1: Preprocessing with MinMaxScaler
We scale the features (GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, Research) between 0 and 1:
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Step 2: Model Architecture for Regression
Since this is a regression task (predicting a continuous value `Chance of Admit` between 0 and 1), the output layer has **1 neuron** with a **Linear activation** function:
```python
model = Sequential()
model.add(Dense(7, activation='relu', input_dim=7))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='linear')) # Output layer for regression
```

### Step 3: Compilation and Evaluation
We compile with **Mean Squared Error (MSE)** loss and evaluate using $R^2$ score:
```python
model.compile(loss='mean_squared_error', optimizer='Adam')
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)

# R2 Evaluation
from sklearn.metrics import r2_score
y_pred = model.predict(X_test_scaled)
print("R2 Score:", r2_score(y_test, y_pred)) # Achieves ~0.77
```
