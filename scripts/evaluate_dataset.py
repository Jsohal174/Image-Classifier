"""
Batch Evaluation Script
Run YOLO detection on entire dataset and save results
"""

import argparse
import os
import sys
from pathlib import Path
import time
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import yaml
from tqdm import tqdm


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


def preprocess_image(image_path, model_image_size):
    """
    Preprocess image for YOLO model

    Args:
        image_path: Path to image file
        model_image_size: Tuple (height, width) to resize to

    Returns:
        image: PIL Image object
        image_data: Preprocessed numpy array
    """
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    resized_image = image.resize(model_image_size, Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension
    return image, image_data


def draw_boxes_on_image(image, boxes, classes, class_names, scores, thickness=2):
    """
    Draw bounding boxes on image

    Args:
        image: PIL Image object
        boxes: numpy array of box coordinates [y_min, x_min, y_max, x_max]
        classes: numpy array of class indices
        class_names: list of class names
        scores: numpy array of confidence scores
        thickness: line thickness for boxes
    """
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        y_min, x_min, y_max, x_max = box
        predicted_class = class_names[int(classes[i])]
        score = scores[i]
        label = f'{predicted_class} {score:.2f}'

        # Draw box
        for j in range(thickness):
            draw.rectangle(
                [x_min + j, y_min + j, x_max - j, y_max - j],
                outline='red'
            )

        # Draw label background
        label_size = draw.textbbox((0, 0), label, font=font)
        label_height = label_size[3] - label_size[1]
        label_width = label_size[2] - label_size[0]

        if y_min - label_height >= 0:
            text_origin = (x_min, y_min - label_height)
        else:
            text_origin = (x_min, y_min + 1)

        draw.rectangle(
            [text_origin[0], text_origin[1],
             text_origin[0] + label_width, text_origin[1] + label_height],
            fill='red'
        )
        draw.text(text_origin, label, fill='white', font=font)

    del draw


def save_detection_results(image_name, boxes, classes, scores, output_file):
    """
    Save detection results to JSON file

    Args:
        image_name: Name of the image file
        boxes: numpy array of box coordinates
        classes: numpy array of class indices
        scores: numpy array of scores
        output_file: Path to output JSON file
    """
    results = {
        'image': image_name,
        'detections': []
    }

    for i in range(len(boxes)):
        detection = {
            'class': int(classes[i]),
            'score': float(scores[i]),
            'bbox': [float(x) for x in boxes[i]]  # [y_min, x_min, y_max, x_max]
        }
        results['detections'].append(detection)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


def evaluate_directory(
    model,
    image_dir,
    output_dir,
    class_names,
    config,
    save_images=True,
    save_json=True
):
    """
    Run evaluation on all images in directory

    Args:
        model: Loaded YOLO model
        image_dir: Directory containing images
        output_dir: Directory to save results
        class_names: List of class names
        config: Configuration dictionary
        save_images: Whether to save annotated images
        save_json: Whether to save detection JSON files

    Returns:
        Dictionary with evaluation statistics
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    # Create output directories
    if save_images:
        img_output_dir = output_dir / 'images'
        img_output_dir.mkdir(parents=True, exist_ok=True)

    if save_json:
        json_output_dir = output_dir / 'detections'
        json_output_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
    image_files = sorted(image_files)

    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return None

    print(f"\nFound {len(image_files)} images")
    print(f"Processing with model...")

    # Statistics
    stats = {
        'total_images': len(image_files),
        'total_detections': 0,
        'detections_per_class': {name: 0 for name in class_names},
        'processing_time': 0,
        'avg_time_per_image': 0
    }

    model_image_size = tuple(config['model']['input_size'])

    start_time = time.time()

    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Load and preprocess
        image, image_data = preprocess_image(img_path, model_image_size)

        # Run detection (placeholder - needs actual YOLO implementation)
        # In real implementation, you would:
        # yolo_outputs = model(image_data)
        # scores, boxes, classes = yolo_eval(yolo_outputs, ...)

        # For now, create dummy detections to show structure
        scores = np.array([])
        boxes = np.array([])
        detected_classes = np.array([])

        # Update statistics
        n_detections = len(scores)
        stats['total_detections'] += n_detections

        for cls in detected_classes:
            class_name = class_names[int(cls)]
            stats['detections_per_class'][class_name] += 1

        # Save annotated image
        if save_images and n_detections > 0:
            draw_boxes_on_image(image, boxes, detected_classes,
                              class_names, scores)
            output_path = img_output_dir / img_path.name
            image.save(output_path, quality=95)

        # Save detection JSON
        if save_json:
            json_path = json_output_dir / f"{img_path.stem}.json"
            save_detection_results(img_path.name, boxes,
                                 detected_classes, scores, json_path)

    # Calculate final statistics
    total_time = time.time() - start_time
    stats['processing_time'] = total_time
    stats['avg_time_per_image'] = total_time / len(image_files)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO model on dataset'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        required=True,
        help='Directory containing images to evaluate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/evaluation',
        help='Directory to save results (default: output/evaluation)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Skip saving annotated images'
    )
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Skip saving detection JSON files'
    )

    args = parser.parse_args()

    print("="*60)
    print("YOLO Dataset Evaluation")
    print("="*60)

    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = load_config(args.config)

    # Load class names
    classes_path = config['model']['classes_path']
    print(f"Loading class names from {classes_path}...")
    class_names = read_classes(classes_path)
    print(f"  ✓ Loaded {len(class_names)} classes")

    # Load model
    print(f"\nLoading model from {config['model']['model_path']}...")
    try:
        model = tf.keras.models.load_model(
            config['model']['model_path'],
            compile=False
        )
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        print("\nNote: Make sure you have downloaded the YOLO model weights.")
        print("See TRAINING_GUIDE.md for instructions.")
        return

    # Run evaluation
    print(f"\nEvaluating on images in {args.dataset_dir}...")
    stats = evaluate_directory(
        model=model,
        image_dir=args.dataset_dir,
        output_dir=args.output_dir,
        class_names=class_names,
        config=config,
        save_images=not args.no_images,
        save_json=not args.no_json
    )

    if stats is None:
        return

    # Print statistics
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Average detections per image: {stats['total_detections']/stats['total_images']:.2f}")
    print(f"\nProcessing time: {stats['processing_time']:.2f}s")
    print(f"Average time per image: {stats['avg_time_per_image']:.3f}s")

    print("\nDetections per class:")
    for class_name, count in stats['detections_per_class'].items():
        if count > 0:
            print(f"  {class_name}: {count}")

    print(f"\nResults saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
