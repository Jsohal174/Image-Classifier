# YOLO Car Detection 🚗

A real-time car detection system using the YOLO (You Only Look Once) algorithm, implemented as part of the Deep Learning Specialization by DeepLearning.AI. This project demonstrates object detection capabilities for autonomous driving applications.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.3+](https://img.shields.io/badge/tensorflow-2.3+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Implementation](#implementation)
- [Results](#results)
- [Configuration](#configuration)
- [References](#references)

## 🎯 Overview

This project implements the YOLO (You Only Look Once) object detection algorithm for detecting cars in images captured from a car-mounted camera. The system is designed for autonomous driving applications where real-time object detection is crucial for safety and navigation.

**Key Highlights:**
- Real-time object detection using YOLOv2
- Detection of 80 different object classes (COCO dataset)
- Non-max suppression for accurate bounding boxes
- Intersection over Union (IoU) for box filtering
- Modular, production-ready code structure

## ✨ Features

- **Fast Detection**: Single forward pass through the network for real-time performance
- **High Accuracy**: Pre-trained on COCO dataset with 80 object classes
- **Configurable Parameters**: Easy configuration via YAML file
- **Modular Design**: Clean separation of concerns with dedicated modules
- **Well-Documented**: Comprehensive documentation and type hints
- **Test Coverage**: Unit tests for core functionality

## 📁 Project Structure

```
yolo-car-detection/
│
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── box_utils.py             # Bounding box utilities (IoU, conversions)
│   ├── yolo_filters.py          # Box filtering and NMS functions
│   └── yolo_detector.py         # Main detector class
│
├── model_data/                   # Model files (not included in repo)
│   ├── yolo.h5                  # Pre-trained YOLO weights
│   ├── coco_classes.txt         # 80 COCO class names
│   └── yolo_anchors.txt         # Anchor box dimensions
│
├── images/                       # Input images directory
├── output/                       # Output images with detections
├── tests/                        # Unit tests
│
├── config.yaml                   # Configuration file
├── detect.py                     # Main detection script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/Jsohal174/yolo-car-detection.git
cd yolo-car-detection
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download model files**

Due to file size constraints, the pre-trained model weights are not included in this repository. You'll need to:

- Download YOLOv2 weights (yolo.h5)
- Download COCO class names (coco_classes.txt)
- Download anchor boxes (yolo_anchors.txt)

Place these files in the `model_data/` directory.

## 💻 Usage

### Basic Usage

Run detection on a single image:

```bash
python detect.py --image images/test.jpg
```

### Advanced Usage

Specify custom configuration:

```bash
python detect.py --image images/test.jpg --config config.yaml --output output/result.jpg
```

### Using as a Library

```python
from src.yolo_detector import yolo_eval
from src.box_utils import yolo_boxes_to_corners
import tensorflow as tf

# Load your model and run detection
# ... (see detect.py for full example)
```

## 🧠 Model Details

### Architecture

- **Input**: Images of shape (608, 608, 3)
- **Output**: Tensor of shape (19, 19, 5, 85)
  - 19×19 grid cells
  - 5 anchor boxes per cell
  - 85 values per box: (p_c, b_x, b_y, b_h, b_w, c_1, ..., c_80)

### Anchor Boxes

The model uses 5 predefined anchor boxes that represent typical object aspect ratios in the training data. These help the model predict boxes of appropriate shapes and sizes.

### Detection Pipeline

1. **Forward Pass**: Image → CNN → (19, 19, 5, 85) encoding
2. **Score Filtering**: Remove boxes with low confidence
3. **Non-Max Suppression**: Eliminate overlapping boxes
4. **Output**: Final set of detected objects with bounding boxes

## 🔧 Implementation

### Core Functions

#### 1. Box Filtering (`yolo_filter_boxes`)
```python
def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6)
```
Filters boxes based on confidence score threshold.

#### 2. Intersection over Union (`iou`)
```python
def iou(box1, box2)
```
Calculates overlap between two bounding boxes.

#### 3. Non-Max Suppression (`yolo_non_max_suppression`)
```python
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5)
```
Removes redundant overlapping boxes.

#### 4. YOLO Evaluation (`yolo_eval`)
```python
def yolo_eval(yolo_outputs, image_shape, max_boxes=10, score_threshold=0.6, iou_threshold=0.5)
```
Complete evaluation pipeline combining all filtering steps.

## 📊 Results

The model successfully detects cars and other objects in driving scenarios with:
- **Detection Speed**: Real-time performance on GPU
- **Accuracy**: High precision on COCO dataset classes
- **Robustness**: Works in various lighting and weather conditions

Example detection output:
```
Found 10 boxes for test.jpg
car    0.89 (367, 300) (745, 648)
car    0.80 (761, 282) (942, 412)
car    0.74 (159, 303) (346, 440)
bus    0.67 (5, 266) (220, 407)
...
```

## ⚙️ Configuration

Edit `config.yaml` to customize detection parameters:

```yaml
detection:
  max_boxes: 10              # Maximum detections per image
  score_threshold: 0.6       # Confidence threshold
  iou_threshold: 0.5         # NMS threshold
```

**Parameters Guide:**
- `score_threshold`: Higher = fewer, more confident detections (range: 0.0-1.0)
- `iou_threshold`: Lower = more aggressive suppression of overlapping boxes (range: 0.0-1.0)
- `max_boxes`: Maximum number of objects to detect per image

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Object detection using deep learning
- ✅ Implementation of YOLO algorithm
- ✅ Non-max suppression techniques
- ✅ Bounding box coordinate systems
- ✅ TensorFlow and Keras usage
- ✅ Computer vision for autonomous driving

## 📚 References

- **YOLO Papers**:
  - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) - Redmon et al., 2016
  - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) - Redmon & Farhadi, 2016

- **Implementation**:
  - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K) - Allan Zelener
  - [Official YOLO Website](https://pjreddie.com/darknet/yolo/)

- **Dataset**:
  - Drive.ai Sample Dataset (provided by [drive.ai](https://www.drive.ai/))
  - Licensed under Creative Commons Attribution 4.0

## 🤝 Contributing

This is a learning project, but suggestions and improvements are welcome! Feel free to:
- Open an issue for bugs or questions
- Submit a pull request for enhancements
- Share your results and experiences

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Jaskirat Singh Sohal**

- GitHub: [@Jsohal174](https://github.com/Jsohal174)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/jaskiratsohal)

## 🙏 Acknowledgments

- DeepLearning.AI for the excellent Deep Learning Specialization course
- Andrew Ng and the course instructors
- Drive.ai for providing the car detection dataset
- The YOLO authors for their groundbreaking work in object detection

---

If you found this project helpful, please consider giving it a star!

**Note**: This project was created as part of the Convolutional Neural Networks course (Course 4) in the Deep Learning Specialization by DeepLearning.AI on Coursera.
