import numpy as np 
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
import os
import torch
import cv2
import torch
from lightglue import LightGlue, SIFT
from lightglue.utils import load_image, rbd, read_image

class SatelliteImageMatcherSIFTLightGlue:
    """
    A class to match satellite images using SIFT and LightGlue.

    Attributes:
        extractor (SIFT): Feature extractor using SIFT.
        lightglue (LightGlue): Matcher using LightGlue.
        feats_1 (dict): Features of the first image.
        feats_2 (dict): Features of the second image.
        kpts_1 (np.ndarray): Keypoints of the first image.
        kpts_2 (np.ndarray): Keypoints of the second image.
        m_kpts_1 (np.ndarray): Matched keypoints from the first image.
        m_kpts_2 (np.ndarray): Matched keypoints from the second image.
        matches_1_2 (dict): Matches between the two images.
        matches (np.ndarray): Matched indices.
        image_1 (np.ndarray): First image.
        image_2 (np.ndarray): Second image.
    """
    def __init__(self, max_num_keypoints=None):
        """
        Initializes the SatelliteImageMatcherSIFTLightGlue class by loading the pre-trained LightBlue model with SIFT exctractor
        """
        self.extractor = SIFT(max_num_keypoints=max_num_keypoints).eval()
        self.lightglue = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1, filter_threshold=0.01).eval()
        self.feats_1 = None
        self.feats_2 = None
        self.kpts_1 = None
        self.kpts_2 = None
        self.m_kpts_1 = None
        self.m_kpts_2 = None
        self.matches_1_2 = None
        self.matches = None
        self.image_1 = None
        self.image_2 = None

    def __normalize_image(self, image):
        """
        Normalize an image for processing.

        Args:
            image (np.ndarray): Input image.

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        if image.ndim == 3:
            return torch.tensor(image.transpose((2, 0, 1)) / 255.0, dtype=torch.float) # to (C, H, W) transpose
        image = image[None]  # add channel axis if grayscale
        return torch.tensor(image / 255.0, dtype=torch.float) 
            
    def match_images(self, image_1, image_2):
        """
        Match two satellite images using SIFT and LightGlue.

        Args:
            image_1 (np.ndarray): First image.
            image_2 (np.ndarray): Second image.
        """
        # Get images for visualization, no need for ploting normalized images
        self.image_1 = image_1
        self.image_2 = image_2
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)
        image_1 = self.__normalize_image(image_1)
        image_2 = self.__normalize_image(image_2)
        feats_1 = self.extractor.extract(image_1)
        feats_2 = self.extractor.extract(image_2)
        matches_1_2 = self.lightglue({"image0": feats_1, "image1": feats_2})
        feats_1, feats_2, matches_1_2 = [
            rbd(x) for x in [feats_1, feats_2, matches_1_2]
        ]  # remove batch dimension
        
        kpts_1, kpts_2, matches = feats_1["keypoints"], feats_2["keypoints"], matches_1_2["matches"]
        m_kpts_1, m_kpts_2 = kpts_1[matches[..., 0]], kpts_2[matches[..., 1]]
        
        self.feats_1 = feats_1
        self.feats_2 = feats_2
        self.kpts_1 = kpts_1
        self.kpts_2 = kpts_2
        self.m_kpts_1 = m_kpts_1
        self.m_kpts_2 = m_kpts_2
        self.matches_1_2 = matches_1_2
        self.matches = matches

    def calculate_inliers(self):
        """
        Calculate inliers using the RANSAC method.

        Returns:
            dict: Dictionary containing inliers, inlier ratio, and fundamental matrix.
        """
        if len(self.m_kpts_1) < 8 or len(self.m_kpts_2) < 8:
            print("Not enough keypoints to compute fundamental matrix. At least 8 points are required.")
            return

        # Ensure keypoints are numpy arrays
        m_kpts_1 = self.m_kpts_1.cpu().numpy()
        m_kpts_2 = self.m_kpts_2.cpu().numpy()
        
        # Use RANSAC method for robust fundamental matrix estimation
        fmat, inliers_mask = cv2.findFundamentalMat(
            m_kpts_1,
            m_kpts_2,
            cv2.USAC_ACCURATE,
            1.0,
            0.99,
            100000
        )
        
        inliers_mask = inliers_mask.flatten().astype(bool)  # Convert to boolean
        inlier_ratio = np.sum(inliers_mask) / len(self.matches)
    
        # Return results
        return {
            "keypoints_0": self.kpts_1,
            "keypoints_1": self.kpts_2,
            "matches": self.matches,
            "inliers": inliers_mask,
            "inlier_ratio": inlier_ratio,
            "fmat": fmat,
        }

    def visualize_keypoints_matches(self):
        """
        Visualize keypoints and matches between two images.
        """
        from lightglue import viz2d
        kpc0, kpc1 = viz2d.cm_prune(self.matches_1_2["prune0"]), viz2d.cm_prune(self.matches_1_2["prune1"])
        viz2d.plot_images([self.image_1, self.image_2])
        viz2d.plot_keypoints([self.kpts_1, self.kpts_2], colors=[kpc0, kpc1], ps=10)
        viz2d.plot_images([self.image_1, self.image_2])
        viz2d.plot_keypoints([self.m_kpts_1, self.m_kpts_2], colors="lime", ps=10)
        axes = viz2d.plot_images([self.image_1, img_2])
        viz2d.plot_matches(self.m_kpts_1, self.m_kpts_2, color="lime", lw=0.2)

if __name__ == '__main__':
    # Example of usage
    matcher_sift_lightglue = SatelliteImageMatcherSIFTLightGlue()
    img_1, img_2 = # Here should be loaded photos in RGB format
    matcher_sift_lightglue.match_images(img_1, img_2)
    inliers = matcher_sift_lightglue.calculate_inliers()
    if inliers is not None:
        print(f"Total Keypoints Image 1: {len(inliers['keypoints_0'])}")
        print(f"Total Keypoints Image 2: {len(inliers['keypoints_1'])}")
        print(f"Total Matches: {len(inliers['matches'])}")
        print(f"Number of Inliers: {np.sum(inliers['inliers']) if inliers['inliers'] is not None else 0}")
        print(f"Inlier Ratio: {inliers['inlier_ratio']:.2f}")
    matcher_sift_lightglue.visualize_keypoints_matches()
