# -*- coding: utf-8 -*-
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn
import time
from PIL import Image
import sys
import io

# …Ë÷√ƒ¨»œ±‡¬ÎŒ™ utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def load_model(model_path):
    """
    Load model from Horizon development board (.bin format)
    """
    models = dnn.load(model_path)
    return models[0]  # Get model object

def prepare_input(image_path, input_shape):
    """
    Prepare input data: load the image and resize it to the specified shape
    Assume the image is RGB and the input format is uint8
    """
    # Open image
    image = Image.open(image_path)
    img_width, img_height = image.size
    target_height, target_width = input_shape[1], input_shape[2]

    # Resize image if needed
    if img_height != target_height or img_width != target_width:
        print(f"Image size {image.size} does not meet requirements, resizing to {input_shape[1]}x{input_shape[2]} ...")
        image = image.resize((target_width, target_height), Image.BILINEAR)

    image = np.array(image)
    if image.ndim == 2:  # If grayscale
        image = np.stack([image] * 3, axis=-1)

    image = image.astype(np.uint8)  # Ensure uint8 type

    # Zero-padding if image is smaller than target
    if image.shape[0] < target_height or image.shape[1] < target_width:
        padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded_image[:image.shape[0], :image.shape[1]] = image
        image = padded_image

    return image

def infer(model, input_data):
    """
    Perform inference
    """
    outputs = model.forward(input_data)
    return outputs

def test_inference_speed(model_path, image_path, input_shape, warmup_runs=10, test_runs=100):
    """
    Test inference speed on the Horizon development board
    """
    model = load_model(model_path)
    input_data = prepare_input(image_path, input_shape)

    for _ in range(warmup_runs):
        infer(model, input_data)

    start_time = time.time()
    for _ in range(test_runs):
        infer(model, input_data)
    end_time = time.time()

    total_time = end_time - start_time
    average_time = total_time / test_runs
    print(f"Inference Speed Test Complete:")
    print(f"Input Shape: {input_shape}")
    print(f"Total Time: {total_time:.4f} seconds, Average Inference Time: {average_time:.6f} seconds")
    print(f"Inferences per second: {test_runs / total_time:.2f}")

if __name__ == "__main__":
    model_path = "/root/Robot_pet/face_ws/face_recog/model_output_dir/retainaface.bin"
    image_path = "/root/Robot_pet/face_ws/face_recog/imgs/5.jpg"
    input_shape = (3, 640, 640)

    test_inference_speed(model_path, image_path, input_shape)
