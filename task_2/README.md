# Satellite Image Matcher with LoFTR

Mmatching keypoints between satellite images using the LoFTR (Local Feature TRansformer) model and SIFT+LightBlue model. This project supports keypoint matching, inlier calculation, and visualization of matches.

---

## Features

- **Keypoint Matching**: Matches keypoints between two satellite images using the pre-trained models.
- **Fundamental Matrix Estimation**: Estimates the fundamental matrix using RANSAC and calculates inliers.
- **Visualization**: Provides visualizations of matched keypoints and inliers between two images.

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA support for faster computations)
- Kornia (for image processing and feature matching)
- OpenCV (for image transformations and matrix estimation)
