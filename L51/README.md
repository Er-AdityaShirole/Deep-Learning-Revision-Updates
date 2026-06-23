# Lesson 51: Pretrained Models & Transfer Learning in CNNs

Welcome to my revision notes for **Lesson 51** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **Pretrained Models**: Why we utilize models pre-trained on massive datasets.
2. **ImageNet & ILSVRC**: The history of large-scale computer vision benchmarks.
3. **Classic Architectures**: AlexNet, VGG16/19, and ResNet50.
4. **Keras Applications API**: Implementing pretrained vision pipelines.

---

## 📝 Key Revision Points

### What is Transfer Learning?
Transfer learning is taking the weights of a model already trained on a massive dataset (like ImageNet with 14 million images and 1,000 classes) and applying them to a new, smaller dataset.
- Early layers in CNNs learn edge detection, color blobs, and basic shapes, which are generic and reusable for almost any computer vision task.
- We keep the pre-trained feature extractor (convolutional base) and only train a new classification head (dense layers) matching our target classes.

### History & Benchmarks
- **ImageNet**: A large visual database organized in a hierarchy.
- **ILSVRC (ImageNet Large Scale Visual Recognition Challenge)**: An annual competition that tracked state-of-the-art vision models.
- **AlexNet (2012)**: The winner of ILSVRC 2012, utilizing GPUs, ReLU, Dropout, and Max Pooling. It proved the dominance of Deep Learning.
- **VGG16 (2014)**: Introduced simple $3 \times 3$ conv layers stacked deeply (16 layers).
- **ResNet50 (2015)**: Introduced residual/skip connections, allowing training of extremely deep networks (50+ layers) without vanishing gradients.

---

## 💻 Code Walkthrough (from screenshots)

Using pre-trained weights in Keras:
```python
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained ResNet50 with ImageNet weights
model = ResNet50(weights='imagenet')

# Load and preprocess sample image
img_path = 'chair.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict and decode top 3 classes
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Returns e.g. [('n031797e1', 'folding_chair', 0.9252)]
```
