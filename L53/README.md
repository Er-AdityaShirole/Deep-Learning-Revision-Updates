# Lesson 53: Transfer Learning in Practice

> Theme: reuse the visual intelligence of a large CNN, then adapt it carefully to a smaller custom dataset.

---

## Quick Map

| Idea | Meaning | When to use |
| :--- | :--- | :--- |
| Feature extraction | Freeze the pretrained convolutional base and train only a new classifier head. | Small dataset, limited compute, target images similar to ImageNet. |
| Data augmentation | Create transformed training views such as rotation, zoom, shift, shear, and horizontal flips. | When overfitting appears because the dataset is small. |
| Fine-tuning | Unfreeze a few top layers of the pretrained base and train them with a very small learning rate. | After the custom head is already trained and stable. |

---

## Core Intuition

A pretrained CNN is not just a classifier. Its convolutional layers are a strong image feature extractor.

Early layers usually learn generic visual primitives: edges, corners, color gradients, and simple textures. Middle and deeper layers combine those primitives into richer patterns such as object parts, repeated textures, and high-level visual concepts.

Transfer learning works because many of these features are useful across image tasks. Instead of learning all filters from scratch, we borrow a trained convolutional base and only teach the model the final decision boundary for our dataset.

---

## Workflow 1: Feature Extraction Without Data Augmentation

Notebook: `transfer_learning_feature_extraction(without_data_augmentation).ipynb`

The basic recipe is:

1. Load a pretrained model such as VGG16 with `include_top=False`.
2. Freeze the convolutional base.
3. Pass images through the base.
4. Add a small custom classification head.
5. Train only the new head.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

conv_base = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(150, 150, 3)
)

conv_base.trainable = False

model = Sequential([
    conv_base,
    Flatten(),
    Dense(256, activation="relu"),
    Dense(1, activation="sigmoid")
])
```

Freezing protects the borrowed representation and keeps training fast. If the dataset is small, updating millions of pretrained parameters can quickly destroy useful ImageNet features.

---

## Workflow 2: Feature Extraction With Data Augmentation

Notebook: `transfer_learning_feature_extraction(data_augmentation)_ipynb.ipynb`

Data augmentation improves generalization by making the model see many reasonable variations of the same image.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
```

Augmentation should usually be applied to the training data only. Validation and test data should represent real untouched examples, apart from deterministic preprocessing such as resizing and scaling.

---

## Workflow 3: Fine-Tuning

Notebook: `transfer_learning_finetuning.ipynb`

Fine-tuning is the second stage. First train the classifier head while the base is frozen. Then unfreeze only the top part of the convolutional base and continue training gently.

```python
conv_base.trainable = True

for layer in conv_base.layers[:-4]:
    layer.trainable = False
```

Lower layers learn very general patterns, so changing them is usually unnecessary and risky. Upper layers are more task-specific, so they can be adjusted to the new dataset.

Fine-tuning should nudge pretrained filters, not rewrite them. A high learning rate can cause catastrophic forgetting.

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(
    optimizer=RMSprop(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

## Revision Checkpoints

- Transfer learning is most useful when the dataset is small and training a deep CNN from scratch would overfit.
- `include_top=False` removes the original ImageNet classifier and keeps the convolutional base.
- Freezing prevents pretrained weights from being updated.
- Data augmentation increases input diversity without collecting new data.
- Fine-tuning should happen after the custom head has learned something useful.
- Only a few deeper layers are usually unfrozen during fine-tuning.
- Use a very small learning rate during fine-tuning.

---

## Common Mistakes

| Mistake | Why it hurts |
| :--- | :--- |
| Fine-tuning from the beginning | The randomly initialized head sends noisy gradients into the pretrained base. |
| Unfreezing the full model on a tiny dataset | Overfitting and forgetting become likely. |
| Using heavy augmentation on validation data | Validation metrics become noisy and less meaningful. |
| Large learning rate during fine-tuning | Useful pretrained filters can be damaged quickly. |

---

## One-Line Memory Hook

Train the new head first, then fine-tune the borrowed eyes very gently.
