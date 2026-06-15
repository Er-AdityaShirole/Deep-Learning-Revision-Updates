# Lesson 11: Customer Churn Prediction using ANN (Keras Code Walkthrough)

Welcome to my revision notes for **Lesson 11** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Practical ANN Implementation using Keras/TensorFlow**.
2. **Data Preprocessing**: Handling categorical variables and feature scaling.
3. **Sequential Model API**: Building layer stacks.
4. **Model Compilation & Training**: Loss functions, optimizers, and epochs.

---

## 💻 Code walkthrough

This lesson uses the notebook [notebook8ad570467f.ipynb](file:///C:/Users/lenovo/Desktop/Deep%20Learning/L11/notebook8ad570467f.ipynb) to predict customer churn.

### Step 1: Preprocessing Data
We drop identifier columns and dummy-encode categorical columns (`Geography` and `Gender`):
```python
df.drop(columns = ['RowNumber','CustomerId','Surname'], inplace=True)
df = pd.get_dummies(df, columns=['Geography','Gender'], drop_first=True)
```

### Step 2: Feature Scaling
ANN convergence is highly sensitive to input scale. We apply `StandardScaler`:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)
```

### Step 3: Model Architecture
We stack dense layers sequentially using Keras:
```python
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense

model = Sequential()
# Input layer (input_dim=11) with a hidden layer of 11 neurons
model.add(Dense(11, activation='sigmoid', input_dim=11))
# Second hidden layer
model.add(Dense(11, activation='sigmoid'))
# Output layer (1 neuron for binary classification)
model.add(Dense(1, activation='sigmoid'))
```

### Step 4: Compile and Fit
We compile with **Adam** optimizer and **Binary Crossentropy** loss:
```python
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_trf, y_train, batch_size=50, epochs=100, validation_split=0.2)
```
