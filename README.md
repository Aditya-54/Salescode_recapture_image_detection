

A lightweight image pipeline for detecting whether an input image is a **real photograph** or a **photograph of a screen (recapture)**.

The project was developed for the SalesCode.ai "Spot the Fake Photo" assignment. Instead of training a deep neural network, the solution relies on handcrafted forensic features and a compact machine learning model designed for fast on-device inference.

## Overview

The detector analyzes physical artifacts introduced when a camera photographs a display. Rather than recognizing image content, it focuses on forensic cues such as periodic screen patterns, texture, gradients, color statistics, glare, and compression artifacts.

The final model is a soft-voting ensemble of Logistic Regression and Gradient Boosting with a model size of approximately **128 KB**.

## Approach

The pipeline extracts handcrafted image forensic features including:

- Moiré and periodicity features from the FFT spectrum
- Laplacian sharpness
- Gradient orientation statistics
- Edge density
- Local Binary Pattern (LBP) texture descriptors
- RGB color moments (mean, standard deviation, skewness)
- HSV saturation statistics
- Highlight and glare estimation
- JPEG blockiness features

These features are combined into a lightweight feature vector and classified using a small ensemble model.

## Dataset

The dataset used in this project was prepared by me using my Android phone camera.

- Source: Google Drive folder - https://drive.google.com/drive/folders/1YEHBTL3Bx4tCa5JbIjPsFB5ikKByn0g_?usp=drive_link
- Collection setup: images were captured directly with an Android phone in real-world conditions.
- Classes:
  - **Real**: natural photos of real-world scenes
  - **Screen recapture**: photos of a display showing content such as webpages, videos, or images


## Results

Evaluation was performed on a dataset collected using a single mobile phone.

- 103 images (51 real, 52 screen recaptures)
- Leave-One-Out Cross Validation Accuracy: **95.15%**
- Repeated 5-Fold Cross Validation Accuracy: **92.5% ± 6%**

The reported numbers reflect performance only on the collected dataset. Performance on unseen devices, cameras, and displays may differ.

## Performance

| Metric | Value |
|--------|------:|
| Model Size | ~128 KB |
| Warm Inference | ~120 ms/image |
| Model Prediction | <1 ms |
| Runtime | CPU Only |

Most of the runtime is spent decoding large JPEG images rather than model inference.




## Running

Install dependencies

```bash
pip install -r requirements.txt
```

Predict a single image

```bash
python predict.py path/to/image.jpg
```





## References

The feature engineering approach is inspired by prior work in image forensics and recaptured image detection.

- Ke, Y., Shan, Q., Qin, L., Min, W., & Sun, M. "Recaptured Image Detection Based on Local Texture Descriptor."
- Cao, H., Kot, A. C., & Zhou, X. "Image Recapture Detection Using Learning-Based Features."
- Lukas, J., Fridrich, J., & Goljan, M. "Digital Camera Identification from Sensor Pattern Noise."
- Ojala, T., Pietikäinen, M., & Mäenpää, T. "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns."
- Gonzalez, R. C., & Woods, R. E. *Digital Image Processing* (FFT, gradients, and frequency analysis).
