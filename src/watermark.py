from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import random

from utils import show_image, preprocess_image, clip_eps, get_label

def generate_adversaries_targeted(image_tensor, delta, model, true_index, target_indices, optimizer,eps):
    """
    Generate adversarial perturbations to embed the watermark.
    
    Args:
        image_tensor (tf.Tensor): Input image tensor.
        delta (tf.Variable): Perturbation tensor.
        model (tf.keras.Model): Pre-trained ImageNet model.
        true_index (int): Index of the true class (e.g., panda).
        target_indices (list): Indices of the target classes (secret labels).
        optimizer (tf.keras.optimizers.Optimizer): Optimizer for training.
    
    Returns:
        tf.Tensor: Final perturbation tensor.
    """
    scc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    for t in range(350):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            inp = preprocess_input(image_tensor + delta)
            predictions = model(inp, training=False)
            
            # Ensure the primary classification remains dominant
            true_loss = scc_loss(tf.convert_to_tensor([true_index]), predictions)
            
            # Encourage the model to increase logits for the target labels
            target_loss = 0
            for label in target_indices:
                target_loss += scc_loss(tf.convert_to_tensor([label]), predictions)
            
            # Combine the losses with appropriate weights
            loss = true_loss + target_loss  # Adjust the weight (0.1) as needed
            
            if t % 20 == 0:
                print(f"Iteration {t}, Loss: {loss.numpy()}")
        
        # Get gradients and update delta
        gradients = tape.gradient(loss, delta)
        optimizer.apply_gradients([(gradients, delta)])
        
        # Clip delta to ensure perturbations are small
        delta.assign_add(clip_eps(delta,eps))
    
    return delta

def perturb_image(image_path, true_label, target_labels, model, optimizer, EPS):
    """
    Perturb an image to embed a watermark and verify the watermark.
    
    Args:
        image_path (str): Path to the input image.
        true_label (int): Index of the true class (e.g., panda).
        target_labels (list): Indices of the target classes (secret labels).
        model (tf.keras.Model): Pre-trained ImageNet model.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer for training.
    """
    # Load and preprocess image
    sample_image = show_image(image_path)
    preprocessed_image = preprocess_image(sample_image)

    # Generate predictions before any adversaries
    unsafe_preds = model.predict(preprocess_input(preprocessed_image))
    print(unsafe_preds)
    unsafe_probs = tf.nn.softmax(unsafe_preds).numpy()  # Apply softmax
    print("Prediction before adv.:", decode_predictions(unsafe_probs, top=3)[0])

    # Initialize the perturbation tensor
    image_tensor = tf.constant(preprocessed_image, dtype=tf.float32)
    delta = tf.Variable(tf.zeros_like(image_tensor), trainable=True)

    # Get the learned delta and display it
    delta_tensor = generate_adversaries_targeted(image_tensor, delta, model, true_label, target_labels, optimizer,EPS)
    plt.imshow(50 * delta_tensor.numpy().squeeze() + 0.5)
    plt.show()

    # See if the image changes
    plt.imshow((image_tensor + delta_tensor).numpy().squeeze() / 255)
    plt.show()

    # Generate prediction
    perturbed_image = preprocess_input(image_tensor + delta_tensor)
    preds = model.predict(perturbed_image)
    print(preds)
    probs = tf.nn.softmax(preds).numpy()  # Apply softmax
    print("Prediction after adv.:", decode_predictions(probs, top=10)[0])

    # Verify the watermark
    if verify_watermark(probs, target_labels, threshold=0.000000001):
        print("Watermark verified!")
    else:
        print("Watermark NOT verified.")

def verify_watermark(probs, target_labels, threshold=0.1):
    """
    Verify the watermark by checking if the probabilities of the target labels exceed a threshold.
    
    Args:
        probs (numpy.ndarray): Probabilities output by the model (shape: [1, 1000]).
        target_labels (list): Indices of the target classes (secret labels).
        threshold (float): Threshold for watermark verification.
    
    Returns:
        bool: True if the watermark is verified, False otherwise.
    """
    for label in target_labels:
        if probs[0, label] < threshold:
            return False  # At least one target label does not exceed the threshold
    return True  # All target labels exceed the threshold