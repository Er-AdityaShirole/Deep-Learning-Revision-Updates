# Lesson 41: CNN vs. Visual Cortex (The Famous Cat Experiment)

Welcome to my revision notes for **Lesson 41** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **The Hubel & Wiesel Experiment (1959)**: The biological inspiration behind CNNs.
2. **Receptive Fields and Feature Extraction**: Simple cells vs. Complex cells in the visual cortex.
3. **History of CNNs**: From Neocognitron to modern ConvNets.

---

## 📝 Key Revision Points

### The Hubel & Wiesel Cat Experiment
In 1959, David Hubel and Torsten Wiesel recorded electrical activity in individual neurons of a cat's visual cortex while showing various light patterns on a screen.

> [!NOTE]
> They discovered that neurons in the primary visual cortex do not fire in response to uniform light. Instead, they fire in response to **local, oriented lines or edges** (horizontal, vertical, diagonal).

```
         Oriented Light Bar -------> Screen -------> Cat Eye -------> Electrode
                                                                        |
                                                                        v
                                                             Specific Neurons Fire!
```

### Simple Cells vs. Complex Cells
- **Simple Cells**: Fire when they see a line in a specific location with a specific orientation (e.g., a vertical bar at the center).
- **Complex Cells**: Also detect orientation, but are robust to the exact position of the bar (translation invariance). They combine inputs from simple cells.

### Impact on CNN History
This biological architecture of starting with simple feature detectors and combining them hierarchically to form complex detectors inspired:
1. **Kunihiko Fukushima (1980 - Neocognitron)**: First computational implementation of S-cells (simple) and C-cells (complex).
2. **Yann LeCun (1989/1998 - LeNet)**: Integrated backpropagation with convolutional layers and shared weights, creating the modern CNN.
