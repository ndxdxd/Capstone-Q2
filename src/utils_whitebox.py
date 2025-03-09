from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import random
from sklearn.metrics import roc_curve, auc
IMAGENET_LABELS = "../data/imagenet_class_index.json"
with open(IMAGENET_LABELS) as f:
    IMAGENET_CLASSES = {int(i): x[1] for i, x in json.load(f).items()}
# Function to load and display an image
def show_image(image_path, show_img = True):
    image_pixels = plt.imread(image_path)
    if show_img:
        plt.imshow(image_pixels)
        plt.show()
    return image_pixels

# Function to preprocess an image for inference
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_image(image_pixels, preprocess=False):
    if preprocess:
        image_pixels = preprocess_input(image_pixels)
    image_pixels = cv2.resize(image_pixels, (224, 224))
    image_pixels = np.expand_dims(image_pixels, axis=0)

    
    return image_pixels

# Clipping utility to project delta back to a favorable pixel range
def clip_eps(delta_tensor, EPS):
    return tf.clip_by_value(delta_tensor, clip_value_min=-EPS, clip_value_max=EPS)

# Parse the label
def get_label(preds, IMAGENET_CLASSES):
    print(IMAGENET_CLASSES[preds.argmax()])

def plot_logit_histograms(all_logits_before, all_logits_after, target_labels, bins=100, save_dir="../plots/"):
    """
    Plot histograms of logit scores before and after watermarking for each target label and save as images.
    
    Args:
        all_logits_before (dict): Logit scores before watermarking for each label.
        all_logits_after (dict): Logit scores after watermarking for each label.
        target_labels (list): Indices of the target classes (secret labels).
        bins (int): Number of bins for the histogram.
        save_dir (str): Directory to save the images.
    """
    for label in target_labels:
        plt.figure(figsize=(10, 6))
        plt.hist(all_logits_before[label], bins=bins, alpha=0.5, label="Before Watermarking")
        plt.hist(all_logits_after[label], bins=bins, alpha=0.5, label="After Watermarking")
        plt.legend()
        plt.xlabel("Logit Score")
        plt.ylabel("Frequency")
        plt.title(f"Logit Scores for Label {IMAGENET_CLASSES[label]} (Index: {label}) at k = 4")
        
        # Save the plot as an image
        filename = f"{save_dir}logit_histogram_label_{label}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory

def plot_combined_histogram(all_logits_before, all_logits_after, bins=100, save_dir="../plots/"):
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
    filename = f"{save_dir}combined_histogram.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

def plot_roc_curve(all_logits_before, all_logits_after, target_labels, save_dir="../plots/"):
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
