# YOLO Object Detection 🚗

A real-time object detection system using YOLOv3 (You Only Look Once) with OpenCV's DNN module. This project demonstrates practical object detection for autonomous driving and computer vision applications.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.0+-green.svg)](https://opencv.org/)
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

This project implements the YOLOv3 (You Only Look Once) object detection algorithm using OpenCV's DNN module for detecting objects in images. The system can identify 80 different object classes from the COCO dataset, making it suitable for autonomous driving and general computer vision applications.

**Key Highlights:**
- Real-time object detection using YOLOv3
- Detection of 80 different object classes (COCO dataset)
- OpenCV DNN module for efficient inference
- Non-max suppression for accurate bounding boxes
- Simple, self-contained implementation
- No TensorFlow/PyTorch dependencies required

## ✨ Features

- **Fast Detection**: Single forward pass through the network for real-time performance
- **High Accuracy**: Pre-trained YOLOv3 model on COCO dataset with 80 object classes
- **Easy to Use**: Simple command-line interface
- **Lightweight**: Uses OpenCV DNN (no heavy deep learning frameworks)
- **Self-Contained**: All detection logic in a single Python script
- **Visual Output**: Annotated images with bounding boxes and confidence scores

## 📁 Project Structure

```
yolo-car-detection/
│
├── model_data/                   # YOLO model files
│   ├── yolov3.weights           # Pre-trained YOLOv3 weights (237 MB)
│   ├── yolov3.cfg               # YOLOv3 network configuration
│   └── coco_classes.txt         # 80 COCO class names
│
├── images/                       # Input images directory
│   ├── example.jpg              # Sample image
│   └── example2.webp            # Sample image
│
├── output/                       # Output images with detections
│   └── detection_result.jpg     # Annotated output images
│
├── detect_real.py                # Main detection script
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

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

4. **Download YOLO model files** (Required - ~240 MB)

Download the YOLOv3 weights and configuration:
```bash
mkdir -p model_data
cd model_data

# Download YOLOv3 weights (237 MB)
wget https://pjreddie.com/media/files/yolov3.weights

# Download YOLOv3 configuration
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg

# Download COCO class names
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O coco_classes.txt

cd ..
```

**Note:** If `wget` is not available, download the files manually from the URLs above and place them in the `model_data/` directory.

## 💻 Usage

### Basic Usage

Run detection on any image:

```bash
python3 detect_real.py --image images/example.jpg
```

This will:
- Load the YOLOv3 model using OpenCV DNN
- Detect objects in the image (80 COCO classes)
- Draw bounding boxes with labels and confidence scores
- Save result to `output/detection_result.jpg`

### Specify Custom Output Location

```bash
python3 detect_real.py --image path/to/your/image.jpg --output output/my_result.jpg
```

### Example Output

```
============================================================
🚗 YOLO REAL-TIME OBJECT DETECTION
============================================================

📷 Loading image: images/example.jpg
   Image size: 1920x1080

🧠 Loading YOLO model...
🔍 Running detection...

✅ DETECTED 3 OBJECTS:
============================================================

   [1] CAR: 94.0% confidence
       Location: x=367, y=300, w=378, h=348
   [2] CAR: 87.0% confidence
       Location: x=761, y=282, w=181, h=130
   [3] PERSON: 82.0% confidence
       Location: x=159, y=303, w=187, h=137

💾 Saved output to: output/detection_result.jpg
============================================================
```

## 🧠 Model Details

### YOLOv3 Architecture

- **Framework**: OpenCV DNN module (no TensorFlow/PyTorch required)
- **Input**: Images resized to 416×416 pixels, normalized to [0-1]
- **Network**: 106 convolutional layers (Darknet-53 backbone)
- **Output**: 3 detection scales for small, medium, and large objects
  - Grid sizes: 13×13, 26×26, 52×52
  - 3 anchor boxes per grid cell
  - 85 values per detection: [x, y, w, h, objectness, 80 class probabilities]

### Detection Classes

The model detects 80 object classes from the COCO dataset, including:
- **Vehicles**: car, bus, truck, motorcycle, bicycle, train, boat, airplane
- **People**: person
- **Animals**: dog, cat, bird, horse, cow, elephant, bear, zebra, giraffe
- **Objects**: traffic light, stop sign, parking meter, bench, backpack, umbrella, handbag
- And many more...

### Detection Pipeline

1. **Image Preprocessing**: Resize to 416×416, normalize pixels, convert BGR→RGB
2. **Forward Pass**: Image → YOLOv3 network → Raw detections from 3 scales
3. **Post-Processing**:
   - Parse detections (extract bounding boxes and class probabilities)
   - Filter by confidence threshold (default: 0.5 or 50%)
   - Apply Non-Maximum Suppression (NMS) to remove duplicate detections
4. **Output**: Final bounding boxes with class labels and confidence scores

## 🔧 Implementation

### How It Works (detect_real.py)

The detection script uses OpenCV's DNN module to run YOLOv3:

#### 1. Load YOLO Model
```python
net = cv2.dnn.readNet("model_data/yolov3.weights", "model_data/yolov3.cfg")
```
Loads the pre-trained YOLOv3 network.

#### 2. Prepare Image
```python
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
```
Converts image to blob format (normalized, resized, color-corrected).

#### 3. Run Detection
```python
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
detections = net.forward(output_layers)
```
Performs forward pass through the network.

#### 4. Parse Detections
```python
for detection in output:
    scores = detection[5:]  # Class probabilities
    class_id = np.argmax(scores)  # Best class
    confidence = scores[class_id]

    if confidence > 0.5:  # Confidence threshold
        # Extract and store bounding box
```
Extracts bounding boxes and filters by confidence.

#### 5. Apply Non-Maximum Suppression
```python
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
```
Removes duplicate/overlapping detections using NMS.

## 📊 Results

The YOLOv3 model successfully detects objects in images with:
- **Detection Speed**: Fast inference using OpenCV DNN (1-3 seconds per image on CPU)
- **Accuracy**: High precision on COCO dataset classes
- **Versatility**: Detects 80 different object classes
- **Robustness**: Works in various lighting and weather conditions

### Performance Characteristics

- **Confidence Threshold**: 0.5 (50%) - Only shows high-confidence detections
- **NMS Threshold**: 0.4 (40% IoU) - Removes overlapping duplicate boxes
- **Output**: Annotated images with colored bounding boxes and labels
- **Console Output**: Detailed detection information with coordinates

## ⚙️ Configuration

### Detection Parameters

The detection parameters are currently hardcoded in `detect_real.py` (lines 88, 102):

```python
# Confidence threshold - minimum confidence to consider a detection
if confidence > 0.5:  # 50% confidence threshold

# Non-Maximum Suppression thresholds
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#                                              ^^^  ^^^
#                                        score_threshold  iou_threshold
```

**To customize detection parameters, edit detect_real.py:**

- **Confidence Threshold** (line 88): Change `0.5` to adjust minimum detection confidence
  - Higher value (e.g., 0.7) = fewer, more confident detections
  - Lower value (e.g., 0.3) = more detections, including less certain ones

- **Score Threshold** (line 102, first `0.5`): Minimum score for NMS
  - Should typically match the confidence threshold

- **IoU Threshold** (line 102, `0.4`): Controls overlap suppression
  - Lower value (e.g., 0.3) = more aggressive suppression (fewer overlapping boxes)
  - Higher value (e.g., 0.6) = allows more overlapping detections

**Note:** The `config.yaml` file exists from a previous implementation but is not currently used by `detect_real.py`.

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Object detection using YOLOv3 algorithm
- ✅ OpenCV DNN module for deep learning inference
- ✅ Non-maximum suppression (NMS) techniques
- ✅ Bounding box coordinate systems and transformations
- ✅ Image preprocessing and normalization
- ✅ Multi-scale object detection
- ✅ Computer vision for autonomous driving and general object detection

## 📚 References

- **YOLO Papers**:
  - [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) - Redmon & Farhadi, 2018
  - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) - Redmon et al., 2016
  - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) - Redmon & Farhadi, 2016

- **Implementation Resources**:
  - [Official YOLO Website](https://pjreddie.com/darknet/yolo/) - Joseph Redmon
  - [OpenCV DNN Module Documentation](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html)
  - [Darknet GitHub Repository](https://github.com/pjreddie/darknet)

- **Dataset**:
  - [COCO Dataset](https://cocodataset.org/) - 80 object classes used for training YOLOv3
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

- Joseph Redmon and Ali Farhadi for creating the YOLO algorithm and YOLOv3
- The OpenCV team for the excellent DNN module
- The COCO dataset team for providing comprehensive object detection training data
- The open-source computer vision community

---

If you found this project helpful, please consider giving it a star! ⭐
