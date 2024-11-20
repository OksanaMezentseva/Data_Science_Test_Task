import os
import cv2
import pickle
import random
import matplotlib.pyplot as plt
from image_loader import ImageLoader
from feature_matcher import FeatureMatcher

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
test_images_path = os.path.join(base_dir, "data", "test_images")
model_path = os.path.join(base_dir, "models", "feature_matcher.pkl")

# Load the saved model
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        matcher = pickle.load(f)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Model not found at {model_path}")

def get_random_images_from_test_images(test_images_path):
    """
    Select two random images from the same tile (folder) in the test images directory.
    """
    # List all tile folders in test images
    tile_folders = [os.path.join(test_images_path, folder) for folder in os.listdir(test_images_path) if os.path.isdir(os.path.join(test_images_path, folder))]
    
    if not tile_folders:
        raise ValueError("No tile folders found in the test images directory.")

    # Randomly select a tile folder
    random_tile_folder = random.choice(tile_folders)
    print(f"Selected tile folder: {os.path.basename(random_tile_folder)}")

    # List all images in the selected folder
    image_files = [os.path.join(random_tile_folder, file) for file in os.listdir(random_tile_folder) if file.endswith("_TCI.jp2")]
    
    if len(image_files) < 2:
        raise ValueError(f"Not enough images in tile folder: {random_tile_folder}")

    # Randomly select two images
    img1_path, img2_path = random.sample(image_files, 2)
    print(f"Selected images: {os.path.basename(img1_path)}, {os.path.basename(img2_path)}")

    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise ValueError("Error loading images.")

    return img1, img2

def run_inference():
    """
    Perform feature matching inference using randomly selected images from the test images.
    """
    # Get two random images from the same tile in test images
    img1, img2 = get_random_images_from_test_images(test_images_path)

    # Perform inference (draw matches)
    print("Running inference...")
    matched_img = matcher.draw_matches(img1, img2)

    # Display results
    plt.figure(figsize=(15, 8))
    plt.imshow(matched_img)
    plt.title("Feature Matching Results")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        print(f"Error: {e}")