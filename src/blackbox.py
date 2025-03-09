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

from src.utils_blackbox import show_image, preprocess_image, clip_eps, get_label

IMAGENET_LABELS = "./data/imagenet_class_index.json"
with open(IMAGENET_LABELS) as f:
    IMAGENET_CLASSES = {int(i): x[1] for i, x in json.load(f).items()}

resnet50 = tf.keras.applications.ResNet50(weights="imagenet", include_top=True, classifier_activation=None)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  
EPS = 0.7 / 255

def generate_adversaries_targeted(image_tensor, delta, model, true_index, target_index, optimizer, eps):
    scc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-3)
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

def verify_watermark(logits_before, logits_after, target_labels, threshold=1):
    """
    Verify the watermark by checking if at least 50% of the target labels have a positive increase in logits.

    Args:
        logits_before (numpy.ndarray): Logits before perturbation (shape: [1, 1000]).
        logits_after (numpy.ndarray): Logits after perturbation (shape: [1, 1000]).
        target_labels (list): Indices of the target classes (secret labels).
        threshold (float): Unused in this version, but kept for potential future extensions.

    Returns:
        bool: True if the watermark is verified, False otherwise.
        float: The average logit difference across target labels.
    """
    logits_d = []
    positive_count = 0
    
    for label in target_labels:
        logit_before = logits_before[0, label]
        logit_after = logits_after[0, label]
        logit_diff = logit_after - logit_before  # Calculate the difference
        
        logits_d.append(logit_diff)
        if logit_diff > 0:
            positive_count += 1
        
        print(f"Label: {label} | Logit Before: {logit_before:.5f} | Logit After: {logit_after:.5f} | Logit Diff: {logit_diff:.5f}")
    
    avg_diff = sum(logits_d) / len(logits_d)
    positive_percentage = (positive_count / len(target_labels)) * 100
    verified = positive_count >= (len(target_labels) / 2)  # Check if at least 50% are positive
    
    print(f"Average Logit Difference: {avg_diff:.5f}")
    print(f"Positive Percentage: {positive_percentage:.2f}%")
    print(f"Verification Status: {'Verified' if verified else 'Not Verified'}")
    
    return verified, avg_diff


def perturb_image(image_path, true_label, target_labels, model, optimizer, eps):
    
    sample_image = show_image(image_path)
    preprocessed_image = preprocess_image(sample_image)
    image_tensor = tf.constant(preprocessed_image, dtype=tf.float32)

    # Initialize total_delta to accumulate perturbations
    total_delta = tf.Variable(tf.zeros_like(image_tensor), trainable=True)


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
    averaged_delta = total_delta / len(target_labels)
    scaled_delta = averaged_delta * tf.sqrt(tf.cast(len(target_labels), tf.float32))

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
    
    # Normalize delta tensor for visualization
    
    delta_image = 50 * scaled_delta.numpy().squeeze() + 0.5  # Scale and shift
    delta_image = np.clip(delta_image, 0, 1)  # Ensure values are in [0,1]
    # Save the image
    output_path = "./output/delta_image.png"
    plt.imsave(output_path, delta_image)  # Use cmap="gray" for better visualization
    print(f"Delta image saved at {output_path}")

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

def perturb_image_2(image_path, true_label, target_labels, model, optimizer, eps, iterations = 350):
    
    sample_image = show_image(image_path)
    preprocessed_image = preprocess_image(sample_image)
    image_tensor = tf.constant(preprocessed_image, dtype=tf.float32)

    # Initialize total_delta to accumulate perturbations
    total_delta = tf.Variable(tf.zeros_like(image_tensor), trainable=True)


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
    averaged_delta = total_delta / len(target_labels)
    scaled_delta = averaged_delta * tf.sqrt(tf.cast(len(target_labels), tf.float32))

    # Apply final perturbation
    perturbed_image = preprocess_input(image_tensor + scaled_delta)
    final_preds = model.predict(perturbed_image)
    final = list(final_preds)
    print("\n=== Logits After Perturbation ===")
    for target in target_labels:
        print(f"Target {IMAGENET_CLASSES[target]}: {final_preds[0][target]:.5f}")

    # Check if watermark is verified
    is_watermarked, avg_diff = verify_watermark(original_logits, final_preds, target_labels)
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

# Return logit scores for secret labels before and after watermarking
    logits_before = {label: original_logits[0, label] for label in target_labels}
    logits_after = {label: final_preds[0, label] for label in target_labels}
    return logits_before,logits_after,original_logits, final_preds

def perturb_image_epsilon_testing(image_path, true_label, target_labels, model, optimizer, eps, save_to):
    sample_image = show_image(image_path)
    preprocessed_image = preprocess_image(sample_image)
    image_tensor = tf.constant(preprocessed_image, dtype=tf.float32)

    # Initialize total_delta to accumulate perturbations
    total_delta = tf.Variable(tf.zeros_like(image_tensor), trainable=True)


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
    averaged_delta = total_delta / len(target_labels)
    scaled_delta = averaged_delta * tf.sqrt(tf.cast(len(target_labels), tf.float32))

    perturbed_image = preprocess_input(image_tensor + scaled_delta)
    final_preds = model.predict(perturbed_image)
    final = list(final_preds)
    print("\n=== Logits After Perturbation ===")
    for target in target_labels:
        print(f"Target {IMAGENET_CLASSES[target]}: {final_preds[0][target]:.5f}")

    # Check if watermark is verified
    is_watermarked, _ = verify_watermark(original_logits, final_preds, target_labels)
    print(f"\nWatermark Verification: {'Success' if is_watermarked else 'Failure'}")

    # Display the perturbed image
    plt.imshow((image_tensor + scaled_delta).numpy().squeeze() / 255)
    plt.title("Perturbed Image")
    plt.show()
    final_image = (image_tensor + scaled_delta).numpy().squeeze()
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)  # Ensure pixel values are in valid range

    # Save the perturbed image
    output_path = save_to
    cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
    print(f"Watermarked image saved as {output_path}")
    return org, final

    
    

