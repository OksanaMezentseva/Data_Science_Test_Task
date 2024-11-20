import os
import cv2
import pickle
import random
from image_loader import ImageLoader
from feature_matcher import FeatureMatcher

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sorted_images_path = os.path.join(base_dir, "data", "sorted_by_tile")
model_path = os.path.join(base_dir, "models", "feature_matcher.pkl")

# Initialize matcher
matcher = FeatureMatcher()

def get_random_images_from_same_tile(sorted_images_path):
    """
    Select two random images from the same tile (folder).
    """
    # List all tile folders
    tile_folders = [os.path.join(sorted_images_path, folder) for folder in os.listdir(sorted_images_path) if os.path.isdir(os.path.join(sorted_images_path, folder))]
    
    if not tile_folders:
        raise ValueError("No tile folders found in the sorted images directory.")

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

def train_model():
    """
    Train the feature matching model using randomly selected images from the same tile.
    """
    # Get two random images from the same tile
    img1, img2 = get_random_images_from_same_tile(sorted_images_path)

    # Train algorithm (find features and match them)
    print("Training feature matcher...")
    kp1, des1, kp2, des2 = matcher.find_features_px(img1, img2)
    good_matches = matcher.compare_features(des1, des2)

    print(f"Number of good matches: {len(good_matches)}")

    # Save the matcher object for reuse
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(matcher, f)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error: {e}")