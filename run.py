import os
import shutil
import kagglehub

# Define dataset name
dataset_name = "titericz/imagenet1k-val"

# Automatically determine the KaggleHub cache directory
cache_dir = os.path.expanduser("~/.cache/kagglehub/datasets")
dataset_path = os.path.join(cache_dir, dataset_name, "versions/1")

# Define destination folder in the current working directory
destination = os.path.abspath("imagenet1k-val")

# Download dataset (KaggleHub saves it in its cache location)
path = kagglehub.dataset_download(dataset_name)

# Ensure the dataset path exists and is not empty
if os.path.exists(dataset_path) and os.listdir(dataset_path):
    # Copy dataset instead of moving to prevent empty folder issues
    shutil.copytree(dataset_path, destination, dirs_exist_ok=True)

    print(f"Dataset successfully copied to: {destination}")

    # Optional: Delete original cache to save space
    shutil.rmtree(dataset_path)
    print(f"Deleted cache folder: {dataset_path}")
else:
    print("Error: Dataset path not found or dataset is empty!")

print(f"Path to dataset files: {destination}")
