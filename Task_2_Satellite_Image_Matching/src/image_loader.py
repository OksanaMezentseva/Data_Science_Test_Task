import cv2
import glob
import os
import numpy as np


class ImageLoader:
    """Loads and preprocesses satellite images."""

    def __init__(self, base_dir: str, pattern: str = "**/*_TCI.jp2"):
        self.base_dir = base_dir
        self.pattern = pattern
        self.image_files = glob.glob(os.path.join(base_dir, self.pattern), recursive=True)

    def load_images(self, max_dimension=1024):
        """Loads and resizes images."""
        images = []
        for image_file in self.image_files:
            img = self._load_image(image_file, max_dimension)
            images.append(img)
        return images

    def _load_image(self, image_path, max_dimension):
        """Reads and resizes an image."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        scale = max_dimension / max(height, width)
        return cv2.resize(img, (int(width * scale), int(height * scale)))