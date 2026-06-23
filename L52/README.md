# Lesson 52: Visualizing CNN Filters & Feature Maps

Welcome to my revision notes for **Lesson 52** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **What does a CNN see?**: Interpreting model parameters and activations.
2. **Visualizing Filter Weights**: Seeing edge and texture detectors.
3. **Visualizing Feature Maps**: Observing intermediate outputs of convolutional layers.
4. **Hierarchical Abstraction**: Watching features transition from simple edges to abstract concepts.

---

## 📝 Key Revision Points

### 1. Visualizing Filter Weights
By plotting the weights of the filters (kernels) in the first convolutional layer, we can see exactly what pattern each filter is looking for. Since first-layer filters have shape $3 \times 3 \times 3$ (RGB), we can normalize their weights between $0$ and $1$ and render them as mini images. They typically show:
- Vertical and horizontal line detectors.
- Color gradient detectors (e.g. green-to-red transitions).

### 2. Visualizing Feature Maps
A feature map is the activation output of a convolutional layer after passing an image through it.
- **Early Layers**: Retain most of the spatial detail of the image. They act like edge filters (e.g. highlighting outline contours).
- **Deep Layers**: Retain very little detail. They become highly abstract and sparse. A single active pixel in a deep feature map might represent the presence of a complex feature (e.g. "dog nose") in that receptive field.

---

## 💻 Code Walkthrough (from `visualising_cnn.ipynb`)

We load VGG16 and inspect feature maps of different layers:

### Step 1: Submodel Extraction for Feature Maps
To get the activation output of the first convolutional layer, we create a submodel:
```python
from keras.applications.vgg16 import VGG16
from keras.models import Model
import matplotlib.pyplot as plt

vgg_model = VGG16()
# Create model that outputs the activations of the first conv layer
visual_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[1].output)
```

### Step 2: Run Prediction and Plot Feature Maps
We pass an image (`kohli.jpg`) and plot all 64 filter activation outputs in a grid:
```python
# features shape is (1, 224, 224, 64)
features = visual_model.predict(image)

fig = plt.figure(figsize=(20, 15))
for i in range(1, features.shape[3] + 1):
    plt.subplot(8, 8, i)
    plt.imshow(features[0, :, :, i-1], cmap='gray')
plt.show()
```

### Step 3: Visualizing across Depths
By picking layers at index `[2, 5, 9, 13, 17]` and plotting their feature maps side-by-side, we observe the visual representation becoming increasingly coarse and abstract, illustrating the hierarchical feature learning of CNNs.
