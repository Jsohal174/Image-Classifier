#!/usr/bin/env python3
"""
Real YOLO Object Detection
Takes an input image, detects objects, outputs image with bounding boxes
"""

import argparse
import sys
from pathlib import Path

def detect_with_opencv(image_path, output_path):
    """Use OpenCV's DNN module for detection"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("❌ OpenCV not installed!")
        print("Install with: pip3 install opencv-python")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("🚗 YOLO REAL-TIME OBJECT DETECTION")
    print(f"{'='*60}\n")

    # Load image
    print(f"📷 Loading image: {image_path}")
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        sys.exit(1)

    height, width = image.shape[:2]
    print(f"   Image size: {width}x{height}")

    # Check for YOLO weights
    weights_path = Path("model_data/yolov3.weights")
    config_path = Path("model_data/yolov3.cfg")

    if not weights_path.exists() or not config_path.exists():
        print(f"\n⚠️  YOLO weights not found!")
        print(f"\nTo download YOLO weights:")
        print(f"   mkdir -p model_data")
        print(f"   cd model_data")
        print(f"   wget https://pjreddie.com/media/files/yolov3.weights")
        print(f"   wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg")
        print(f"\nFor now, using SIMPLE DEMO MODE with mock detections...\n")

        # Create demo output with mock detections
        create_demo_output(image, image_path, output_path)
        return

    # Load YOLO
    print(f"\n🧠 Loading YOLO model...")
    net = cv2.dnn.readNet(str(weights_path), str(config_path))

    # Load class names
    classes_path = Path("model_data/coco_classes.txt")
    if classes_path.exists():
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    else:
        classes = ["object"]  # Default

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Run detection
    print(f"🔍 Running detection...")
    detections = net.forward(output_layers)

    # Process detections
    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    print(f"\n✅ DETECTED {len(indices)} OBJECTS:")
    print(f"{'='*60}\n")

    # Draw boxes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]) if class_ids[i] < len(classes) else "object"
            confidence = confidences[i]
            color = colors[class_ids[i]]

            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Draw label
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            print(f"   [{i+1}] {label.upper()}: {confidence:.1%} confidence")
            print(f"       Location: x={x}, y={y}, w={w}, h={h}")
    else:
        print("   No objects detected with confidence > 0.5")

    # Save output
    cv2.imwrite(str(output_path), image)
    print(f"\n💾 Saved output to: {output_path}")
    print(f"{'='*60}\n")


def create_demo_output(image, image_path, output_path):
    """Create demo output with sample detections"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Please install opencv-python: pip3 install opencv-python")
        sys.exit(1)

    height, width = image.shape[:2]

    # Create some demo detections based on image size
    demo_detections = [
        {"label": "car", "confidence": 0.94, "x": int(width*0.3), "y": int(height*0.4),
         "w": int(width*0.2), "h": int(height*0.25), "color": (0, 0, 255)},
        {"label": "car", "confidence": 0.87, "x": int(width*0.6), "y": int(height*0.5),
         "w": int(width*0.15), "h": int(height*0.2), "color": (255, 0, 0)},
    ]

    print(f"\n✅ DEMO MODE - DETECTED {len(demo_detections)} OBJECTS:")
    print(f"{'='*60}\n")

    for i, det in enumerate(demo_detections, 1):
        # Draw rectangle
        cv2.rectangle(image,
                     (det['x'], det['y']),
                     (det['x'] + det['w'], det['y'] + det['h']),
                     det['color'], 3)

        # Draw label background
        text = f"{det['label']}: {det['confidence']:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        cv2.rectangle(image,
                     (det['x'], det['y'] - text_height - 10),
                     (det['x'] + text_width, det['y']),
                     det['color'], -1)

        # Draw label text
        cv2.putText(image, text,
                   (det['x'], det['y'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        print(f"   [{i}] {det['label'].upper()}: {det['confidence']:.1%} confidence")
        print(f"       Location: x={det['x']}, y={det['y']}, w={det['w']}, h={det['h']}")

    # Add watermark
    cv2.putText(image, "YOLO Detection Demo",
               (10, height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save output
    cv2.imwrite(str(output_path), image)
    print(f"\n💾 Saved output to: {output_path}")
    print(f"\n💡 TIP: Download real YOLO weights for actual detection!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Object Detection - Detect objects in images'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/detection_result.jpg',
        help='Path to save output image (default: output/detection_result.jpg)'
    )

    args = parser.parse_args()

    # Check input image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run detection
    detect_with_opencv(image_path, output_path)


if __name__ == '__main__':
    main()
