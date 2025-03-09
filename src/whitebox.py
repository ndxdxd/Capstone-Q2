from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import random
import os
from PIL import Image

from utils_whitebox import show_image, preprocess_image, clip_eps, get_label
IMAGENET_LABELS = "../data/imagenet_class_index.json"
with open(IMAGENET_LABELS) as f:
    IMAGENET_CLASSES = {int(i): x[1] for i, x in json.load(f).items()}

def generate_adversaries_targeted(image_tensor, delta, model, true_index, target_indices, optimizer,eps,iterations = 350):
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
    
    for t in range(iterations):
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
            target_loss /= len(target_indices)
            
            loss = true_loss + target_loss * (3.0)  # Adjust the weight as needed
            
            
            if t % 20 == 0:
                print(f"Iteration {t}, Loss: {loss.numpy()}")
        
        # Get gradients and update delta
        gradients = tape.gradient(loss, delta)
        optimizer.apply_gradients([(gradients, delta)])
        
        # Clip delta to ensure perturbations are small
        delta.assign_add(clip_eps(delta,eps))
    
    return delta

def perturb_image(image_path, true_label, target_labels, model, optimizer, EPS ,iterations = 350, save_dir="../output"):
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sample_image = show_image(image_path)
    preprocessed_image = preprocess_image(sample_image)

    preprocessed_path = os.path.join(save_dir, "preprocessed_image.png")
    plt.imsave(preprocessed_path, preprocessed_image.squeeze())
    print(f"Preprocessed image saved at: {preprocessed_path}")


    # Generate predictions before any adversaries
    unsafe_preds = model.predict(preprocess_input(preprocessed_image))
    print("Logits before adv.:", decode_predictions(unsafe_preds, top=3)[0])

    true_label = np.argmax(unsafe_preds, axis=1)[0]
    print(f"True label (max logit): {IMAGENET_CLASSES[true_label]} (Index: {true_label})")

    print("\nPredictions for secret labels BEFORE perturbation:")
    for label in target_labels:
        label_name = IMAGENET_CLASSES[label]
        logit = (unsafe_preds[0, label])
        print(f"Label: {label_name} (Index: {label}), Logit: {logit}")
        
    # Initialize the perturbation tensor
    image_tensor = tf.constant(preprocessed_image, dtype=tf.float32)
    delta = tf.Variable(tf.zeros_like(image_tensor), trainable=True)

    # Get the learned delta and display it
    delta_tensor = generate_adversaries_targeted(image_tensor, delta, model, true_label, target_labels, optimizer, EPS, iterations)

    delta_image = 50 * delta_tensor.numpy().squeeze() + 0.5  # Scale and shift
    delta_image = np.clip(delta_image, 0, 1)  # Ensure values are in [0,1]
    
    # Save the image
    output_path = "../output/delta_image.png"
    plt.imsave(output_path, delta_image)  # Use cmap="gray" for better visualization
    print(f"Delta image saved at {output_path}")


    plt.imshow(50 * delta_tensor.numpy().squeeze() + 0.5)
    plt.show()


    final_image = (image_tensor + delta_tensor).numpy().squeeze()
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)  # Ensure pixel values are in valid range
    
    # Save the perturbed image
    output_folder = "../output"
    output_path = os.path.join(output_folder, "whitebox_watermarked_image.jpg")  # Save in the output folder
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the image
    cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
    print(f"Watermarked image saved as {output_path}")

    # Clip the pixel values to the valid range [0, 1]
    # perturbed_image = np.clip(perturbed_image, 0, 1)

    # perturbed_image_path = os.path.join(save_dir, "perturbed_image.png")
    # plt.imsave(perturbed_image_path, perturbed_image)
    # print(f"Perturbed image saved at: {perturbed_image_path}")

    # See if the image changes
    
    plt.imshow((image_tensor + delta_tensor).numpy().squeeze() / 255)
    plt.show()

    

    # Generate prediction
    perturbed_image = preprocess_input(image_tensor + delta_tensor)    
    preds = model.predict(perturbed_image)

    


    print("Logits after adv.:", decode_predictions(preds, top=3)[0])

    # Print predictions for the secret labels
    print("\nPredictions for secret labels:")
    for label in target_labels:
        label_name = IMAGENET_CLASSES[label]
        logit = (preds[0, label])
        print(f"Label: {label_name} (Index: {label}), Logit: {logit}")


    # Verify the watermark
    # Verify the watermark using logit differences
    verified, avg_diff = verify_watermark(unsafe_preds, preds, target_labels, threshold=1)
    if verified:
        print("Watermark verified!")
    else:
        print("Watermark NOT verified.")

    # Return logit scores for secret labels before and after watermarking
    logits_before = {label: unsafe_preds[0, label] for label in target_labels}
    logits_after = {label: preds[0, label] for label in target_labels}
    return logits_before,logits_after,unsafe_preds, preds, avg_diff

def verify_watermark(logits_before, logits_after, target_labels, threshold=1):
    """
    Verify the watermark by checking if the increase in average logits of the target labels exceeds a threshold.
    
    Args:
        logits_before (numpy.ndarray): Logits before perturbation (shape: [1, 1000]).
        logits_after (numpy.ndarray): Logits after perturbation (shape: [1, 1000]).
        target_labels (list): Indices of the target classes (secret labels).
        threshold (float): Threshold for the increase in logits.
    
    Returns:
        bool: True if the watermark is verified, False otherwise.
    """
    verified = False  # Assume verification fails unless proven otherwise
    logits_d = []
    positive_changes = 0  # Counter for positive logit differences

    for label in target_labels:
        logit_before = logits_before[0, label]
        logit_after = logits_after[0, label]
        logit_diff = logit_after - logit_before  # Calculate the difference
        
        # Store the logit difference
        logits_d.append(logit_diff)
        print(f"Label: {IMAGENET_CLASSES[label]} ({label}) | Logit Before: {logit_before:.5f} | Logit After: {logit_after:.5f} | Logit Diff: {logit_diff:.5f}")
        
        # Count how many times the logit difference is positive
        if logit_diff > 0:
            positive_changes += 1
    
    # Calculate the percentage of labels with positive logit differences
    positive_percentage = (positive_changes / len(target_labels)) * 100
    print(f"Percentage of labels with positive logit difference: {positive_percentage:.2f}%")
    
    # If 50% or more of the logit differences are positive, consider the watermark verified
    if positive_percentage >= 50:
        verified = True
    
    return verified, logits_d


def display_one_image_per_folder(root_folder, secret_labels,resnet50, optimizer, EPS, iterations = 350, max_folders=10):
    """
    Display one image from each of `max_folders` randomly selected subfolders with the folder name as the title.

    Args:
        root_folder (str): Path to the folder containing subfolders with images.
        max_folders (int): Maximum number of folders to process.
    """
    # Get a list of all subfolders
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    


    subfolders = subfolders[:max_folders]
    

    all_logits_before = {label: [] for label in secret_labels}
    all_logits_after = {label: [] for label in secret_labels}
    count = 1

    # Iterate through each subfolder
    for folder in subfolders:
        # Get the folder name (ID)
        folder_name = os.path.basename(folder)

        # Get a list of images in the folder
        images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]

        print(f'Image: #{count}')
        count += 1
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
            logits_before, logits_after,logits_before_all,logits_after_all, _ = perturb_image(first_image_path, true_label, secret_labels, resnet50, optimizer, EPS, iterations)

            # Store logit scores
            for label in secret_labels:
                all_logits_before[label].append(logits_before[label])
                all_logits_after[label].append(logits_after[label])

        

        else:
            print(f"No images found in folder: {folder_name}")
    return all_logits_before, all_logits_after, logits_before_all,logits_after_all 


def perturb_image_test_epsilon(image_path, true_label, target_labels, model, optimizer, EPS, save_image_to, iterations=350, save_dir="../output/testing_epsilon/"):
    """
    Perturb an image to embed a watermark and verify the watermark.
    
    Args:
        image_path (str): Path to the input image.
        true_label (int): Index of the true class (e.g., panda).
        target_labels (list): Indices of the target classes (secret labels).
        model (tf.keras.Model): Pre-trained ImageNet model.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer for training.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load and preprocess image
    sample_image = show_image(image_path)
    preprocessed_image = preprocess_image(sample_image)

    preprocessed_path = os.path.join("../output", "preprocessed_image.png")
    plt.imsave(preprocessed_path, preprocessed_image.squeeze())
    print(f"Preprocessed image saved at: {preprocessed_path}")

    # Generate predictions before any adversaries
    unsafe_preds = model.predict(preprocess_input(preprocessed_image))
    print("Logits before adv.:", decode_predictions(unsafe_preds, top=3)[0])

    true_label = np.argmax(unsafe_preds, axis=1)[0]
    print(f"True label (max logit): {IMAGENET_CLASSES[true_label]} (Index: {true_label})")

    print("\nPredictions for secret labels BEFORE perturbation:")
    for label in target_labels:
        label_name = IMAGENET_CLASSES[label]
        logit = (unsafe_preds[0, label])
        print(f"Label: {label_name} (Index: {label}), Logit: {logit}")
        
    # Initialize the perturbation tensor
    image_tensor = tf.constant(preprocessed_image, dtype=tf.float32)
    delta = tf.Variable(tf.zeros_like(image_tensor), trainable=True)

    # Generate adversarial perturbation
    delta_tensor = generate_adversaries_targeted(image_tensor, delta, model, true_label, target_labels, optimizer, EPS, iterations)
    plt.imshow(50 * delta_tensor.numpy().squeeze() + 0.5)
    plt.show()

    # Compute final perturbed image
    final_image = (image_tensor + delta_tensor).numpy().squeeze()
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)  # Ensure pixel values are in the valid range
    
    output_path = os.path.join(save_dir, save_image_to)  # Save in the specified directory   
    # Save the image
    cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    print(f"Watermarked image saved at: {output_path}")

    # Display perturbed image
    plt.imshow((image_tensor + delta_tensor).numpy().squeeze() / 255)
    plt.show()

    # Generate prediction on perturbed image
    perturbed_image = preprocess_input(image_tensor + delta_tensor)    
    preds = model.predict(perturbed_image)

    print("Logits after adv.:", decode_predictions(preds, top=3)[0])

    # Print predictions for the secret labels
    print("\nPredictions for secret labels:")
    for label in target_labels:
        label_name = IMAGENET_CLASSES[label]
        logit = (preds[0, label])
        print(f"Label: {label_name} (Index: {label}), Logit: {logit}")

    # Verify the watermark
    verified, avg_diff = verify_watermark(unsafe_preds, preds, target_labels, threshold=0.01)
    if verified:
        print("Watermark verified!")
    else:
        print("Watermark NOT verified.")

    # Return logit scores for secret labels before and after watermarking
    logits_before = {label: unsafe_preds[0, label] for label in target_labels}
    logits_after = {label: preds[0, label] for label in target_labels}
    return logits_before, logits_after, unsafe_preds, preds, avg_diff 



