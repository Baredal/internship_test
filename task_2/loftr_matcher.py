import numpy as np 
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
import os
import torch
import kornia as K
from kornia_moons.viz import draw_LAF_matches
import cv2
import torch

class SatelliteImageMatcherLOFTR:
    """
    A class for matching keypoints between satellite images using the LoFTR model.
    The class supports keypoint matching, inlier calculation, and visualization.

    Attributes:
        matcher (K.feature.LoFTR): The LoFTR model for feature matching.
        image_1 (np.ndarray): The first input image.
        image_2 (np.ndarray): The second input image.
        kpts_1 (np.ndarray): Keypoints from the first image after matching.
        kpts_2 (np.ndarray): Keypoints from the second image after matching.
    """
    def __init__(self):
        """
        Initializes the SatelliteImageMatcherLOFTR class by loading the pre-trained LoFTR model.
        """
        self.matcher = K.feature.LoFTR(pretrained='outdoor').eval()
        self.image_1 = None
        self.image_2 = None
        self.kpts_1 = None
        self.kpts_2 = None

    def __normalize_image(self, image):
        """
        Normalize an image to a PyTorch tensor with shape (C, H, W) and scale the pixel values to [0, 1].

        Args:
            image (np.ndarray): The input image.

        Returns:
            torch.Tensor: The normalized image tensor.
        """
        image = K.utils.image_to_tensor(image).float() / 255.0
        return image.unsqueeze(0)

    def match_images(self, image_1, image_2, confidence=0.9):
        """
        Perform keypoint matching between two images using LoFTR.

        Args:
            image_1 (np.ndarray): The first input image.
            image_2 (np.ndarray): The second input image.
            confidence (float, optional): Confidence threshold for selecting keypoints. Default is 0.9.

        Updates:
            self.kpts_1 (np.ndarray): Keypoints from the first image.
            self.kpts_2 (np.ndarray): Keypoints from the second image.
        """
        # Convert images to tensors and grayscale
        self.image_1 = image_1
        self.image_2 = image_2
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)
        image_1 = self.__normalize_image(image_1)
        image_2 = self.__normalize_image(image_2)
        
        with torch.inference_mode():
            matches = self.matcher({'image0': image_1, 'image1': image_2})

        # Select key points that have a confidence greater than confidence_min
        mask = matches['confidence'] > confidence
        indices = torch.nonzero(mask, as_tuple=True)               
        keypoints_1 = matches['keypoints0'][indices].cpu().numpy()
        keypoints_2 = matches['keypoints1'][indices].cpu().numpy()
        confidence = matches['confidence'][indices].cpu().numpy()

        self.kpts_1 = keypoints_1
        self.kpts_2 = keypoints_2
        
    def calculate_inliers(self):
        """
        Calculate inliers and estimate the fundamental matrix using RANSAC.

        Returns:
            dict: A dictionary containing the fundamental matrix, inlier ratio,
                  inliers mask, keypoints, and the input images.
        """
        if len(self.kpts_1) < 8 or len(self.kpts_2) < 8:
            print("Not enough keypoints to compute fundamental matrix. At least 8 points are required.")
            return
        
        # Use RANSAC method for robust fundamental matrix estimation
        fmat, inliers_mask = cv2.findFundamentalMat(
            self.kpts_1,
            self.kpts_2,
            cv2.USAC_ACCURATE,
            1.0,
            0.99,
            100000
        )
        
        inliers_mask = inliers_mask.astype(bool)  # Convert to boolean
        inlier_ratio = sum(inliers_mask)[0] / len(inliers_mask)
            
        return {
            'image_1' : self.image_1,
            'image_2' : self.image_2,
            'keypoints_1': self.kpts_1,
            'keypoints_2': self.kpts_2,
            'inliers': inliers_mask,
            'fmat': fmat,
            'inlier_ratio': inlier_ratio
        } 

    def draw_matches(self, matches):
        """
        Visualize keypoint matches between the two input images.

        Args:
            matches (dict): A dictionary containing keypoints, inliers, and images,
                            typically returned by `calculate_inliers`.
        """
        image_1 = self.__normalize_image(matches['image_1'])
        image_2 = self.__normalize_image(matches['image_2'])
        draw_LAF_matches(
            K.feature.laf_from_center_scale_ori(
                torch.from_numpy(matches['keypoints_1']).view(1, -1, 2),
                torch.ones(matches['keypoints_1'].shape[0]).view(1, -1, 1, 1),
                torch.ones(matches['keypoints_1'].shape[0]).view(1, -1, 1),
            ),
            K.feature.laf_from_center_scale_ori(
                torch.from_numpy(matches['keypoints_2']).view(1, -1, 2),
                torch.ones(matches['keypoints_2'].shape[0]).view(1, -1, 1, 1),
                torch.ones(matches['keypoints_2'].shape[0]).view(1, -1, 1),
            ),
            torch.arange(matches['keypoints_1'].shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(image_1),
            K.tensor_to_image(image_2),
            matches['inliers'],
            draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
        )
if __name__ == '__main__':
  # Example of usage
  matcher_loftr = SatelliteImageMatcherLOFTR()
  img_1, img_2 = # Here should be photos loaded in RGB format
  matcher_loftr.match_images(img_1, img_2)
  matches = matcher_loftr.calculate_inliers()
  if matches is not None:
      print(f"Total Keypoints Image 1: {len(matches['keypoints_1'])}")
      print(f"Total Keypoints Image 2: {len(matches['keypoints_2'])}")
      print(f"Number of Inliers: {np.sum(matches['inliers']) if matches['inliers'] is not None else 0}")
      print(f"Inlier Ratio: {matches['inlier_ratio']}")
      matcher_loftr.draw_matches(matches)
