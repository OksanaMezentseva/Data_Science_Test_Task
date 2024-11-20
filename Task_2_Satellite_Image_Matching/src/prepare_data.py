import os
import shutil

def sort_tci_images_by_tile(source_directory, destination_directory):
    """
    Sort and copy TCI images by tile.
    
    Args:
        source_directory (str): The directory containing the original images.
        destination_directory (str): The directory to store sorted images.
    """
    # Ensure the source directory exists
    if not os.path.exists(source_directory):
        raise FileNotFoundError(f"Source directory not found: {source_directory}")

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    found_files = False  # Flag to check if any TCI images are found

    # Traverse the source directory recursively
    for root, _, files in os.walk(source_directory):
        for file in files:
            if file.endswith("_TCI.jp2"):  # Select only True Color Images
                found_files = True

                # Extract tile name (e.g., T36UYA)
                tile_name = file.split("_")[0]  

                # Define the tile-specific directory
                tile_dir = os.path.join(destination_directory, tile_name)

                # Create tile directory if it doesn't exist
                if not os.path.exists(tile_dir):
                    os.makedirs(tile_dir)

                source_path = os.path.join(root, file)
                destination_path = os.path.join(tile_dir, file)

                # Copy file instead of moving
                shutil.copy2(source_path, destination_path)

    if not found_files:
        print("No TCI images found in the source directory.")
    else:
        print(f"TCI images successfully copied and sorted by tile into {destination_directory}")


if __name__ == "__main__":
    # Relative paths for the Jupyter Notebook
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    source_directory = os.path.join(current_dir, "../data/images")
    destination_directory = os.path.join(current_dir, "../data/sorted_by_tile")

    # Sort images by tile
    sort_tci_images_by_tile(source_directory, destination_directory)