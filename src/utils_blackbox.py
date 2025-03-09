from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import random
import random
from PIL import Image
from sklearn.metrics import roc_curve, auc
import os


# Function to load and display an image
def show_image(image_path):
    image_pixels = plt.imread(image_path)
    return image_pixels

# Function to preprocess an image for inference
def preprocess_image(image_pixels, preprocess=False):
    if preprocess:
        image_pixels = preprocess_input(image_pixels)
    image_pixels = cv2.resize(image_pixels, (224, 224))
    image_pixels = np.expand_dims(image_pixels, axis=0)
    plt.imshow(image_pixels[0])  # Remove batch dimension before displaying
    plt.show()
    return image_pixels

# Clipping utility to project delta back to a favorable pixel range
def clip_eps(delta_tensor, EPS):
    return tf.clip_by_value(delta_tensor, clip_value_min=-EPS, clip_value_max=EPS)

# Parse the label
def get_label(preds, IMAGENET_CLASSES):
    print(IMAGENET_CLASSES[preds.argmax()])

def display_one_image_per_folder(root_folder, max_folders=10):
    """
    Display one image from each of `max_folders` randomly selected subfolders with the folder name as the title.

    Args:
        root_folder (str): Path to the folder containing subfolders with images.
        max_folders (int): Maximum number of folders to process.
    """
    # Get a list of all subfolders
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    


    subfolders = subfolders[:max_folders]
    secret_labels = [726, 264, 428, 190]

    all_logits_before = {label: [] for label in secret_labels}
    all_logits_after = {label: [] for label in secret_labels}

    # Iterate through each subfolder
    for folder in subfolders:
        # Get the folder name (ID)
        folder_name = os.path.basename(folder)

        # Get a list of images in the folder
        images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]

        if images:
            first_image_path = os.path.join(folder, images[0])

            # Open and display the image
            img = show_image(first_image_path)
            preprocessed_image = preprocess_image(img, preprocess=True)

            preds = resnet50.predict(preprocessed_image)
            print("Logits:", decode_predictions(preds, top=3)[0])
            print("Class idx:", preds.argmax())

            true_label = preds.argmax()
            # Perturb the image and get logit scores
            logits_before, logits_after,logits_before_all,logits_after_all = perturb_image_2(first_image_path, true_label, secret_labels, resnet50, optimizer, EPS, 100)

            # Store logit scores
            for label in secret_labels:
                all_logits_before[label].append(logits_before[label])
                all_logits_after[label].append(logits_after[label])

        

        else:
            print(f"No images found in folder: {folder_name}")
    return all_logits_before, all_logits_after, logits_before_all,logits_after_all 

def plot_combined_histogram(all_logits_before, all_logits_after, bins=100, save_dir="./plots/"):
    """
    Plot a combined histogram of logit scores for all labels before and after watermarking and save as an image.
    
    Args:
        all_logits_before (list): Logit scores for all labels before watermarking.
        all_logits_after (list): Logit scores for all labels after watermarking.
        bins (int): Number of bins for the histogram.
        save_dir (str): Directory to save the image.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(all_logits_before, bins=bins, alpha=0.5, label="Before Watermarking")
    plt.hist(all_logits_after, bins=bins, alpha=0.5, label="After Watermarking")
    plt.legend()
    plt.xlabel("Logit Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Logit Scores for All Labels")

    # Save the plot as an image
    filename = f"{save_dir}blackbox_combined_histogram.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

def plot_roc_curve(all_logits_before, all_logits_after, target_labels, save_dir="./plots/"):
    """
    Plot ROC curve for each target label to determine the optimal threshold and save as an image.
    
    Args:
        all_logits_before (dict): Logit scores before watermarking for each label.
        all_logits_after (dict): Logit scores after watermarking for each label.
        target_labels (list): Indices of the target classes (secret labels).
        save_dir (str): Directory to save the images.
    """
    for label in target_labels:
        # Combine logits before and after watermarking
        y_true = np.concatenate([np.zeros_like(all_logits_before[label]), np.ones_like(all_logits_after[label])])
        y_scores = np.concatenate([all_logits_before[label], all_logits_after[label]])

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Label {IMAGENET_CLASSES[label]} (Index: {label})')
        plt.legend(loc="lower right")

        # Save the plot as an image
        filename = f"{save_dir}roc_curve_label_{label}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory
def plot_logit_histograms(all_logits_before, all_logits_after, target_labels):
    """
    Plot histograms of logit scores before and after watermarking for each target label.
    
    Args:
        all_logits_before (dict): Logit scores before watermarking for each label.
        all_logits_after (dict): Logit scores after watermarking for each label.
        target_labels (list): Indices of the target classes (secret labels).
    """
    IMAGENET_LABELS = "./data/imagenet_class_index.json"
    with open(IMAGENET_LABELS) as f:
        IMAGENET_CLASSES = {int(i): x[1] for i, x in json.load(f).items()}
    for label in target_labels:
        plt.figure(figsize=(10, 6))
        plt.hist(all_logits_before[label], bins=10, alpha=0.5, label="Before Watermarking")
        plt.hist(all_logits_after[label], bins=10, alpha=0.5, label="After Watermarking")
        plt.legend()
        plt.xlabel("Logit Score")
        plt.ylabel("Frequency")
        plt.title(f"Logit Scores for Label {IMAGENET_CLASSES[label]} (Index: {label}) at k = 4")
        plt.show()
        filename = f"./plots/blackbox_logit_dih{label}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory

