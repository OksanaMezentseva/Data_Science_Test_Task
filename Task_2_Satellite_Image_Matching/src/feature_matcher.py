import cv2
import numpy as np

class FeatureMatcher:
    """
    A class for feature detection and matching using SIFT and BFMatcher.
    """

    def __init__(self):
        """
        Initialize SIFT detector and BFMatcher.
        """
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize the image if its values exceed the uint8 range (0-255).

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Normalized image.
        """
        if img.max() > 255:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)

    def find_features_px(self, img1: np.ndarray, img2: np.ndarray):
        """
        Detect features in two images and compute descriptors.

        Args:
            img1 (np.ndarray): First input image.
            img2 (np.ndarray): Second input image.

        Returns:
            tuple: Keypoints and descriptors for both images.
        """
        # Normalize images
        img1 = self._normalize_image(img1)
        img2 = self._normalize_image(img2)

        # Convert to grayscale
        if len(img1.shape) > 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if len(img2.shape) > 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # Detect keypoints and compute descriptors
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        return kp1, des1, kp2, des2

    def compare_features(self, des1, des2, threshold=0.75):
        """
        Match descriptors using the BFMatcher and apply a distance threshold.

        Args:
            des1: Descriptors from the first image.
            des2: Descriptors from the second image.
            threshold (float): Distance threshold for good matches.

        Returns:
            list: List of good matches.
        """
        matches = self.bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < threshold * n.distance]
        return good_matches

    def draw_matches(self, img1, img2, threshold=0.75):
        """
        Visualize matches between two images.

        Args:
            img1 (np.ndarray): First input image.
            img2 (np.ndarray): Second input image.
            threshold (float): Distance threshold for good matches.

        Returns:
            np.ndarray: Image showing the matches.
        """
        kp1, des1, kp2, des2 = self.find_features_px(img1, img2)
        good_matches = self.compare_features(des1, des2, threshold)

        # Draw matches
        matched_img = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return matched_img