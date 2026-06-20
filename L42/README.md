# Lesson 42: The Convolution Operation

Welcome to my revision notes for **Lesson 42** of the *100 Days of Deep Learning* course by CampusX.

---

## 📚 Topics Covered

1. **What is a Convolution?**: Sliding kernels over images.
2. **Edge Detection**: Sobel filters for horizontal and vertical edges.
3. **Convolutions on RGB Images**: Handling multiple color channels.
4. **Multiple Filters**: Creating multi-channel feature maps.

---

## 📝 Key Revision Points

### The Math of 2D Convolution
Convolution involves placing a small filter/kernel (e.g., $3 \times 3$) over an image patch, computing the element-wise product, and summing the results. The filter slides across the image to construct a **Feature Map**.

$$\text{Output Size} = (N - F + 1) \times (N - F + 1)$$
Where $N$ is input size, and $F$ is filter size (assuming stride 1, no padding).

```
Input (6x6) convolved with Filter (3x3) results in Output (4x4)
```

### Edge Detection Example (Sobel Filter)
Consider an image with a vertical edge (left side white [255], right side black [0]). Convolving it with a vertical filter:

$$\text{Vertical Filter} = \begin{bmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{bmatrix}$$

- As the filter slides over the uniform white area, the output is $0$.
- As it slides over the transition boundary, the output is positive (detecting the edge).
- As it slides over the uniform black area, the output is $0$.

### Convolutions on RGB Images (3D Convolutions)
An RGB image has shape $W \times H \times 3$.
- The filter **MUST** have the same number of channels: $f \times f \times 3$.
- Operation: We multiply the filter's red channel with the image's red channel, blue with blue, green with green, and sum all $3 \times f \times f$ products to get **one single scalar value**.
- Output of $(6 \times 6 \times 3) * (3 \times 3 \times 3)$ is a **$4 \times 4 \times 1$** feature map!

```
     Input Image [6x6x3]  *  Filter [3x3x3]  ===>  Output Map [4x4x1]
```

- To detect $K$ different features (e.g., horizontal edges, vertical edges, color borders), we use $K$ different filters. The output becomes a $4 \times 4 \times K$ volume.
