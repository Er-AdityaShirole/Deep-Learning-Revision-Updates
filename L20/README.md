# Lesson 20: Gradient Descent Variants

Welcome to my revision notes for **Lesson 20** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Gradient Descent Refresher**: Updating parameters in the opposite direction of the gradient.
2. **Batch vs Stochastic vs Mini-Batch GD**: Understanding how many samples are used per update.
3. **Batch Size in Keras**: Comparing very small and larger batch sizes in an ANN.
4. **Code Experiment**: Training on the Social Network Ads dataset.

---

## Code Walkthrough

This lesson uses the notebook [Batch_vs_stochastic_GD.ipynb](file:///C:/Users/lenovo/Desktop/Deep%20Learning/L20/Batch_vs_stochastic_GD.ipynb).

### Step 1: Data Preparation
The notebook uses `Age` and `EstimatedSalary` as input features and `Purchased` as the binary target:

```python
df = df[['Age','EstimatedSalary','Purchased']]
X = df.iloc[:,0:2]
y = df.iloc[:,-1]
```

The features are scaled before training:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Step 2: ANN Architecture
A small binary classification neural network is created:

```python
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=2))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

### Step 3: Stochastic Gradient Descent Style
Using `batch_size=1` updates weights after every single row:

```python
history = model.fit(X_scaled, y, epochs=500, batch_size=1, validation_split=0.2)
```

This gives frequent updates but can be noisy.

### Step 4: Batch / Mini-Batch Comparison
A larger batch size performs fewer updates per epoch:

```python
history = model.fit(X_scaled, y, epochs=10, batch_size=250, validation_split=0.2)
```

The key comparison is the tradeoff between noisy but frequent updates and smoother but less frequent updates.

### Why Batch Sizes are Often Powers of 2
Batch sizes like `2, 4, 8, 32, 64, 128, 256` are commonly used because they are RAM/GPU friendly and can be computationally efficient.
