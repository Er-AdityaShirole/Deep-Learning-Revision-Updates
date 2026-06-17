# Lesson 21: Improving Neural Network Performance

Welcome to my revision notes for **Lesson 21** of the *100 Days of Deep Learning* course by CampusX.

---

## Topics Covered

1. **Hyperparameter Tuning**: Improving the model by tuning training and architecture choices.
2. **Common Problems**: Vanishing/exploding gradients, less data, slow training, and overfitting.
3. **Solution Map**: Connecting each training problem with practical deep learning techniques.

---

## Key Revision Points

### Two Broad Ways to Improve a Neural Network
1. Tune neural network hyperparameters.
2. Solve the specific problem that is limiting model performance.

### Common Problems and Solutions

#### Vanishing / Exploding Gradients
Possible solutions:
- Better weight initialization.
- Better activation functions.
- Batch normalization.
- Gradient clipping.

#### Not Enough Data
Possible solutions:
- Transfer learning.
- Unsupervised pretraining.
- Data augmentation where applicable.

#### Slow Training
Possible solutions:
- Better optimizers such as Adam.
- Learning rate scheduling.
- Choosing a suitable batch size.

#### Overfitting
Possible solutions:
- L1 and L2 regularization.
- Dropout layers.
- Early stopping.
- Reducing model complexity or increasing data.

This lecture acts like a roadmap for the next set of techniques used to make neural networks train better and generalize better.
