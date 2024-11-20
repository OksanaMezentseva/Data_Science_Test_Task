import os
import cv2
import pickle
import random
from image_loader import ImageLoader
from feature_matcher import FeatureMatcher

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sorted_images_path = os.path.join(os.path.dirname(base_dir), "data/sorted_by_tile")
model_path = os.path.join(os.path.dirname(base_dir), "models", "feature_matcher.pkl")

# Initialize matcher
matcher = FeatureMatcher()

def train_model_on_multiple_pairs():
    """
    Train the feature matching model using multiple image pairs from different tiles.
    """
    tile_folders = [os.path.join(sorted_images_path, folder) for folder in os.listdir(sorted_images_path) if os.path.isdir(os.path.join(sorted_images_path, folder))]

    if not tile_folders:
        raise ValueError("No tile folders found in the sorted images directory.")

    all_good_matches = []

    for tile_folder in tile_folders:
        print(f"Processing tile folder: {os.path.basename(tile_folder)}")

        # List all images in the folder
        image_files = [os.path.join(tile_folder, file) for file in os.listdir(tile_folder) if file.endswith("_TCI.jp2")]

        if len(image_files) < 2:
            print(f"Not enough images in tile folder: {tile_folder}")
            continue

        # Randomly select pairs of images (e.g., 3 pairs per folder)
        num_pairs = min(3, len(image_files) // 2)
        for _ in range(num_pairs):
            img1_path, img2_path = random.sample(image_files, 2)

            # Load images
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None or img2 is None:
                print(f"Error loading images: {img1_path}, {img2_path}")
                continue

            # Train algorithm on this pair
            kp1, des1, kp2, des2 = matcher.find_features_px(img1, img2)
            good_matches = matcher.compare_features(des1, des2)

            print(f"Number of good matches for this pair: {len(good_matches)}")
            all_good_matches.append(len(good_matches))

    # Save the matcher object for reuse
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(matcher, f)

    print(f"Model saved to {model_path}")
    print(f"Total pairs processed: {len(all_good_matches)}")
    print(f"Average number of good matches: {sum(all_good_matches) / len(all_good_matches) if all_good_matches else 0}")

if __name__ == "__main__":
    try:
        train_model_on_multiple_pairs()
    except Exception as e:
        print(f"Error: {e}")