# Satellite Image Matcher with SIFT-LightGlue and LoFTR

Mmatching keypoints between satellite images using the LoFTR (Local Feature TRansformer) model and SIFT+LightGlue model. This project supports keypoint matching, inlier calculation, and visualization of matches.

---

## Features

- **Keypoint Matching**: Matches keypoints between two satellite images using the pre-trained models.
- **Fundamental Matrix Estimation**: Estimates the fundamental matrix using RANSAC and calculates inliers.
- **Visualization**: Provides visualizations of matched keypoints and inliers between two images.

---

## Notes
To run the installed models, you need to install them locally from the official repository. You will also need to download the dataset from the Kaggle. That's why I attach a link to a demo notebook with the already installed libraries, inference, results and visualizations of the algorithms. To run it you should open it in Kaggle notebook and set up as input Kaggle dataset (deforestation) <br>
[Demo notebook](https://drive.google.com/file/d/1x4tqyyZrD15vhBoGjSQegbd3TSs9C_th/view?usp=sharing)
[Satellite Dataset](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine) <br>
[SIFT-LightGlue](https://github.com/cvg/LightGlue/tree/main) <br>
[LoFTR](https://github.com/zju3dv/LoFTR)

