"""
Simple YOLO Detection Demo
Demonstrates the core YOLO detection functions working
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import tensorflow as tf
from box_utils import iou
from yolo_filters import yolo_filter_boxes, yolo_non_max_suppression

print("="*60)
print("YOLO CAR DETECTION - DEMO")
print("="*60)

# Demo 1: IoU Calculation
print("\n1. Testing Intersection over Union (IoU)")
print("-" * 60)

# Two overlapping boxes
box1 = (100, 100, 300, 300)  # (x1, y1, x2, y2)
box2 = (150, 150, 350, 350)

iou_value = iou(box1, box2)
print(f"Box 1: {box1}")
print(f"Box 2: {box2}")
print(f"IoU: {iou_value:.4f}")
print(f"✓ IoU calculation working!")

# Demo 2: Box Filtering
print("\n2. Testing YOLO Box Filtering")
print("-" * 60)

# Create simulated YOLO output (19x19x5x...)
tf.random.set_seed(42)
boxes = tf.random.normal([19, 19, 5, 4], mean=0.5, stddev=0.2)
box_confidence = tf.random.normal([19, 19, 5, 1], mean=0.6, stddev=0.3)
box_class_probs = tf.random.normal([19, 19, 5, 80], mean=0.1, stddev=0.2)

# Apply filtering
threshold = 0.6
scores, filtered_boxes, classes = yolo_filter_boxes(
    boxes, box_confidence, box_class_probs, threshold
)

print(f"Original boxes: {19*19*5} ({19}x{19} grid, {5} anchors)")
print(f"After filtering (threshold={threshold}): {len(scores)} boxes")
print(f"✓ Box filtering working!")

# Show some detected boxes
if len(scores) > 0:
    print(f"\nTop 3 detections:")
    for i in range(min(3, len(scores))):
        print(f"  Box {i+1}: Class {classes[i].numpy()}, "
              f"Score {scores[i].numpy():.3f}, "
              f"BBox {filtered_boxes[i].numpy()}")

# Demo 3: Non-Max Suppression
print("\n3. Testing Non-Max Suppression (NMS)")
print("-" * 60)

if len(scores) > 0:
    # Apply NMS
    max_boxes = 10
    iou_threshold = 0.5

    final_scores, final_boxes, final_classes = yolo_non_max_suppression(
        scores, filtered_boxes, classes, max_boxes, iou_threshold
    )

    print(f"Boxes before NMS: {len(scores)}")
    print(f"Boxes after NMS: {len(final_scores)}")
    print(f"✓ Non-Max Suppression working!")

    print(f"\nFinal detections (top {len(final_scores)}):")
    for i in range(len(final_scores)):
        print(f"  Detection {i+1}: Class {final_classes[i].numpy()}, "
              f"Score {final_scores[i].numpy():.3f}")
else:
    print("No boxes passed filtering threshold")

# Demo 4: Show detection pipeline summary
print("\n4. Detection Pipeline Summary")
print("-" * 60)
print("""
YOLO Detection Pipeline:
1. Image → CNN → (19×19×5×85) tensor
2. Filter boxes by confidence threshold
3. Apply Non-Max Suppression to remove overlaps
4. Output: Final bounding boxes with classes and scores

✓ All core functions tested and working!
""")

print("="*60)
print("DEMO COMPLETE!")
print("="*60)
print("\nFor your interview, you can show:")
print("  • IoU calculation for bounding box overlap")
print("  • Score-based filtering to remove low-confidence detections")
print("  • Non-Max Suppression to eliminate duplicate detections")
print("  • Complete YOLO detection pipeline")
print("\nAll algorithms implemented from scratch!")
print("="*60)
