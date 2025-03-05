from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import random
import os

from utils import show_image, preprocess_image, clip_eps, get_label
IMAGENET_LABELS = "./data/imagenet_class_index.json"
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
            
            loss = true_loss + target_loss * (1.0)  # Adjust the weight as needed
            
            
            if t % 20 == 0:
                print(f"Iteration {t}, Loss: {loss.numpy()}")
        
        # Get gradients and update delta
        gradients = tape.gradient(loss, delta)
        optimizer.apply_gradients([(gradients, delta)])
        
        # Clip delta to ensure perturbations are small
        delta.assign_add(clip_eps(delta,eps))
    
    return delta

def perturb_image(image_path, true_label, target_labels, model, optimizer, EPS, iterations = 350, save_dir="./output"):
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
    plt.imshow(50 * delta_tensor.numpy().squeeze() + 0.5)
    plt.show()


    final_image = (image_tensor + delta_tensor).numpy().squeeze()
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)  # Ensure pixel values are in valid range
    
    # Save the perturbed image
    output_folder = "./output"
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
    verified, avg_diff = verify_watermark(unsafe_preds, preds, target_labels, threshold=0.01)
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
    print(f"Average Difference: {avg_diff}")
    return verified, avg_diff
    # for label in target_labels:
    #     logit_before = logits_before[0, label]
    #     logit_after = logits_after[0, label]
    #     logit_diff = logit_after - logit_before

    #     if logit_before > logit_after:
    #         return False
        
    #     # If the increase in logits is less than the threshold, verification fails
    #     print(f" Label: {label} , logit_diff : {logit_diff}")
    #     if logit_diff < threshold:
    #         return False
    
    # # If all target labels exceed the threshold, verification succeeds
    # return True


def collect_logits(image_folder, target_labels, model, optimizer, EPS):
    before_logits = []
    after_logits = []

    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(".JPEG"):
                image_path = os.path.join(root, file)
                unsafe_preds, preds = perturb_image(image_path, target_labels, model, optimizer, EPS)
                before_logits.append(unsafe_preds)
                after_logits.append(preds)

    return before_logits, after_logits

