# Setup Instructions

This guide will help you set up the YOLO Car Detection project on your local machine.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Jsohal174/yolo-car-detection.git
cd yolo-car-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
python tests/test_box_utils.py
```

## Required Model Files

This project requires pre-trained YOLO model files that are not included in the repository due to their size. You need to obtain:

### 1. YOLO Model Weights (`yolo.h5`)
- **Size**: ~200 MB
- **Format**: Keras HDF5 format
- **Source**: Original YOLOv2 weights converted to Keras
- **Location**: Place in `model_data/yolo.h5`

### 2. COCO Class Names (`coco_classes.txt`)
- **Content**: List of 80 object class names (one per line)
- **Location**: Place in `model_data/coco_classes.txt`
- **Sample content**:
```
person
bicycle
car
motorbike
...
```

### 3. Anchor Boxes (`yolo_anchors.txt`)
- **Content**: Comma-separated anchor box dimensions
- **Location**: Place in `model_data/yolo_anchors.txt`
- **Format**: `width1,height1,width2,height2,...`

## Directory Structure After Setup

```
yolo-car-detection/
├── model_data/
│   ├── yolo.h5              # ← Add this file
│   ├── coco_classes.txt     # ← Add this file
│   └── yolo_anchors.txt     # ← Add this file
├── images/
│   └── test.jpg             # ← Add test images here
├── output/                   # Detection results will be saved here
└── ...
```

## Configuration

Edit `config.yaml` to customize detection parameters:

```yaml
detection:
  max_boxes: 10              # Maximum number of detections per image
  score_threshold: 0.6       # Confidence threshold (0.0-1.0)
  iou_threshold: 0.5         # NMS threshold (0.0-1.0)
```

## Running Detection

### Basic Usage
```bash
python detect.py --image images/test.jpg
```

### With Custom Config
```bash
python detect.py --image images/test.jpg --config config.yaml
```

### Specify Output Location
```bash
python detect.py --image images/test.jpg --output output/result.jpg
```

## Testing

Run the unit tests to verify installation:

```bash
# Run all tests
python tests/test_box_utils.py

# Expected output:
# Running Box Utils Tests...
# ==================================================
# ✓ test_iou_intersecting_boxes passed
# ✓ test_iou_non_intersecting_boxes passed
# ✓ test_iou_boxes_touching_at_vertices passed
# ✓ test_iou_boxes_touching_at_edges passed
# ✓ test_iou_less_than_one passed
# ==================================================
# All tests passed! ✓
```

## Troubleshooting

### Issue: TensorFlow not installing
**Solution**: Make sure you have Python 3.7-3.9. TensorFlow 2.x may have compatibility issues with Python 3.10+.

```bash
python --version  # Should show 3.7.x, 3.8.x, or 3.9.x
```

### Issue: Model file not found
**Solution**: Ensure `yolo.h5` is in the `model_data/` directory and the path in `config.yaml` is correct.

### Issue: Import errors
**Solution**: Make sure you activated the virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Out of memory errors
**Solution**: Reduce batch size or use a smaller model. You can also try running on CPU only:

```python
# Add to detect.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
```

## GPU Support (Optional)

For faster inference, install TensorFlow with GPU support:

```bash
# Instead of regular tensorflow
pip install tensorflow-gpu==2.3.0

# Requires CUDA Toolkit and cuDNN
```

Check GPU availability:
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

## Next Steps

1. ✅ Set up the environment
2. ✅ Download model files
3. ✅ Add test images to `images/` directory
4. ✅ Run detection on test images
5. ✅ Customize configuration as needed
6. ✅ Explore the code and experiment!

## Need Help?

- Check the [README.md](README.md) for more details
- Open an issue on GitHub
- Review the original notebook for algorithm details

## Additional Resources

- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
