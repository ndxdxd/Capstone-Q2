from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import random
import os
import sys

from src.utils import show_image, preprocess_image, clip_eps, get_label

IMAGENET_LABELS = "./data/imagenet_class_index.json"
with open(IMAGENET_LABELS) as f:
    IMAGENET_CLASSES = {int(i): x[1] for i, x in json.load(f).items()}

resnet50 = tf.keras.applications.ResNet50(weights="imagenet", include_top=True, classifier_activation=None)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  
EPS = 0.1 / 255

def generate_adversaries_targeted(image_tensor, delta, model, true_index, target_index, optimizer, eps):
    scc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    for t in range(100):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            inp = preprocess_input(image_tensor + delta)
            predictions = model(inp, training=False)
            loss = (scc_loss([true_index], predictions) + scc_loss([target_index], predictions))
            
        
        gradients = tape.gradient(loss, delta)
        optimizer.apply_gradients([(gradients, delta)])
        delta.assign_add(clip_eps(delta, eps))
    
    return delta

def verify_watermark(logits_before, logits_after, target_labels, threshold=3):
    """
    Verify the watermark by checking if the increase in logits of the target labels exceeds a threshold.

    Args:
        logits_before (numpy.ndarray): Logits before perturbation (shape: [1, 1000]).
        logits_after (numpy.ndarray): Logits after perturbation (shape: [1, 1000]).
        target_labels (list): Indices of the target classes (secret labels).
        threshold (float): Threshold for the increase in logits.

    Returns:
        bool: True if the watermark is verified, False otherwise.
    """
    verified = True  # Assume verification succeeds until proven otherwise
    logits_d = []
    for label in target_labels:
        logit_before = logits_before[0, label]
        logit_after = logits_after[0, label]
        logit_diff = logit_after - logit_before  # Calculate the difference

        logits_d.append(logit_diff)
        print(f"Label: {IMAGENET_CLASSES[label]} ({label}) | Logit Before: {logit_before:.5f} | Logit After: {logit_after:.5f} | Logit Diff: {logit_diff:.5f}")
    avg_diff = sum(logits_d)/ len(logits_d)
    if avg_diff<threshold:
        verified = False
    print(avg_diff)
    return verified/

def perturb_image(image_path, true_label, model, optimizer, eps, k=5):
    sample_image = show_image(image_path)
    preprocessed_image = preprocess_image(sample_image)
    image_tensor = tf.constant(preprocessed_image, dtype=tf.float32)

    # Initialize total_delta to accumulate perturbations
    total_delta = tf.Variable(tf.zeros_like(image_tensor), trainable=True)

    # Select random target labels
    target_labels = random.sample(list(IMAGENET_CLASSES.keys()), k)

    # Get logits before perturbation
    original_logits = model.predict(preprocess_input(image_tensor))
    org = list(original_logits)
    print("\n=== Logits Before Perturbation ===")
    for target in target_labels:
        print(f"Target {IMAGENET_CLASSES[target]}: {original_logits[0][target]:.5f}")

    # Apply perturbations and accumulate deltas
    for target in target_labels:
        print(f"\nApplying perturbation for target: {IMAGENET_CLASSES[target]} ({target})")
        delta = generate_adversaries_targeted(image_tensor, total_delta, 
                                              model, true_label, target, optimizer, eps)
        total_delta.assign_add(delta)  # Accumulate perturbations

    # Average deltas and scale by sqrt(k)
    averaged_delta = total_delta / k
    scaled_delta = averaged_delta * tf.sqrt(tf.cast(k, tf.float32))
    print(scaled_delta)
    # clipped_delta = tf.clip_by_value(scaled_delta, -eps, eps)
    # print(clipped_delta)
    # Apply final perturbation
    perturbed_image = preprocess_input(image_tensor + scaled_delta)
    final_preds = model.predict(perturbed_image)
    final = list(final_preds)
    print("\n=== Logits After Perturbation ===")
    for target in target_labels:
        print(f"Target {IMAGENET_CLASSES[target]}: {final_preds[0][target]:.5f}")

    # Check if watermark is verified
    is_watermarked = verify_watermark(original_logits, final_preds, target_labels)
    print(f"\nWatermark Verification: {'Success' if is_watermarked else 'Failure'}")

    # Display the perturbed image
    plt.imshow((image_tensor + scaled_delta).numpy().squeeze() / 255)
    plt.title("Perturbed Image")
    plt.show()
    final_image = (image_tensor + scaled_delta).numpy().squeeze()
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)  # Ensure pixel values are in valid range

    # Save the perturbed image
    output_path = "blackbox_watermarked_image.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
    print(f"Watermarked image saved as {output_path}")
    return org, final
    

