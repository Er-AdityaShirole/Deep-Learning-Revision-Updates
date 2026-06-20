# Lesson 39: Hyperparameter Tuning in Deep Learning

Welcome to my revision notes for **Lesson 39** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **The Evolution of Tuning**: Grid Search, Random Search, Bayesian Optimization, Optuna, and ASHA.
2. **Grid Search vs. Random Search**: Why Grid Search is obsolete and why Random Search is superior.
3. **Bayesian Optimization & Optuna**: The industry-standard approach to hyperparameter search.
4. **ASHA (Asynchronous Successive Halving Algorithm)**: State-of-the-art compute-saving pruning algorithm.
5. **Population Based Training (PBT)**: Dynamic tuning for large models.

---

## 📝 Key Revision Points

### 1. Grid Search vs. Random Search
- **Grid Search**: Exhaustively tests every combination in a pre-defined grid. It wastes compute because many hyperparameters (like activation functions or initialization methods) have minimal impact on final loss compared to learning rate.
- **Random Search**: Randomly samples configurations. It is mathematically proven (by James Bergstra) to find optimal configurations faster because it tests unique values across the important dimensions.

```
Grid Search: Wastes trials testing different values of unimportant parameters.
Random Search: Explores unique values of important parameters in every trial.
```

### 2. Bayesian Optimization (Optuna)
Instead of searching blindly, Bayesian Optimization builds a probabilistic model of the objective function (history of previous trials) and chooses the next set of hyperparameters to test by maximizing an **Acquisition Function**.
- It balances **Exploration** (trying unknown regions) and **Exploitation** (searching near good historical points).
- **Optuna** is the most popular library in 2026 for this, utilizing the Tree-structured Parzen Estimator (TPE) algorithm.

### 3. ASHA Pruning (Asynchronous Successive Halving)
ASHA is an early-stopping strategy:
1. Start training many configurations for a very short period (e.g., 2 epochs).
2. Rank the models by validation accuracy.
3. Stop/kill the bottom 75% of models.
4. Let the top 25% continue training for more epochs.
5. Repeat.
This provides **massive compute savings** by avoiding training bad models to completion.

---

## 💻 Code Walkthrough (from `main.ipynb`)

In this lesson, we prepare data (`diabetes.csv`) and scale it before performing tuning with `keras-tuner`:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

df = pd.read_csv('diabetes.csv')
# Preprocessing and correlation analysis show SkinThickness and BloodPressure are negligible
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Scale features before feeding to the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
