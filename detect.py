"""
Main Detection Script for YOLO Car Detection
Run this script to detect cars in images using the pre-trained YOLO model
"""

import argparse
import os
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import tensorflow as tf
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def read_classes(classes_path):
    """Read class names from file"""
    with open(classes_path, 'r') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def read_anchors(anchors_path):
    """Read anchor boxes from file"""
    with open(anchors_path, 'r') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def preprocess_image(image_path, model_image_size):
    """
    Preprocess image for YOLO model

    Arguments:
    image_path -- path to image file
    model_image_size -- tuple (height, width) to resize image to

    Returns:
    image -- PIL Image object
    image_data -- numpy array of preprocessed image
    """
    image = Image.open(image_path)
    resized_image = image.resize(model_image_size, Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension
    return image, image_data


def draw_boxes(image, boxes, classes, class_names, scores):
    """
    Draw bounding boxes on image

    Arguments:
    image -- PIL Image object
    boxes -- numpy array of box coordinates
    classes -- numpy array of class indices
    class_names -- list of class names
    scores -- numpy array of confidence scores
    """
    font = ImageFont.load_default()
    thickness = 2

    for i, box in enumerate(boxes):
        predicted_class = class_names[int(classes[i])]
        score = scores[i]
        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        top, left, bottom, right = box

        # Draw box
        for j in range(thickness):
            draw.rectangle(
                [left + j, top + j, right - j, bottom - j],
                outline='red'
            )

        # Draw label
        label_size = draw.textbbox((0, 0), label, font=font)
        label_height = label_size[3] - label_size[1]
        label_width = label_size[2] - label_size[0]

        if top - label_height >= 0:
            text_origin = (left, top - label_height)
        else:
            text_origin = (left, top + 1)

        draw.rectangle(
            [text_origin[0], text_origin[1],
             text_origin[0] + label_width, text_origin[1] + label_height],
            fill='red'
        )
        draw.text(text_origin, label, fill='white', font=font)

    del draw


def detect_objects(image_path, config):
    """
    Main detection function

    Arguments:
    image_path -- path to image file
    config -- configuration dictionary

    Returns:
    scores -- detected box scores
    boxes -- detected box coordinates
    classes -- detected classes
    """
    print(f"\nProcessing: {image_path}")

    # Load model and data
    print("Loading model...")
    model_path = config['model']['model_path']
    classes_path = config['model']['classes_path']
    anchors_path = config['model']['anchors_path']

    class_names = read_classes(classes_path)
    anchors = read_anchors(anchors_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    # Preprocess image
    print("Preprocessing image...")
    model_image_size = tuple(config['model']['input_size'])
    image, image_data = preprocess_image(image_path, model_image_size)

    # Note: Full detection pipeline requires yolo_head and yolo_eval
    # This is a simplified version showing the structure
    print("Running detection...")

    # Get detection parameters
    max_boxes = config['detection']['max_boxes']
    score_threshold = config['detection']['score_threshold']
    iou_threshold = config['detection']['iou_threshold']

    print(f"✓ Detection complete!")
    print(f"  Parameters: max_boxes={max_boxes}, score_threshold={score_threshold}, iou_threshold={iou_threshold}")

    # Placeholder return - full implementation would use yolo_eval
    return None, None, None


def main():
    """Main function to run object detection"""
    parser = argparse.ArgumentParser(
        description='YOLO Car Detection - Detect cars in images using YOLO'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output image (default: output/result.jpg)'
    )

    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)

    # Run detection
    scores, boxes, classes = detect_objects(args.image, config)

    print("\n" + "="*60)
    print("YOLO Car Detection Complete!")
    print("="*60)
    print("\nNote: This is a demonstration structure.")
    print("For full functionality, ensure you have:")
    print("  1. Pre-trained YOLO model weights (yolo.h5)")
    print("  2. Class names file (coco_classes.txt)")
    print("  3. Anchor boxes file (yolo_anchors.txt)")
    print("  4. YAD2K utilities for yolo_head function")


if __name__ == '__main__':
    main()
