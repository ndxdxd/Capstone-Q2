from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import random

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

def plot_combined_histogram(all_logits_before, all_logits_after, bins=100):
    """
    Plot a combined histogram of logit scores for all labels before and after watermarking.
    Args:
        all_logits_before (list): Logit scores for all labels before watermarking.
        all_logits_after (list): Logit scores for all labels after watermarking.
        bins (int): Number of bins for the histogram.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(all_logits_before, bins=bins, alpha=0.5, label="Before Watermarking")
    plt.hist(all_logits_after, bins=bins, alpha=0.5, label="After Watermarking")
    plt.legend()
    plt.xlabel("Logit Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Logit Scores for All Labels")
    plt.show()