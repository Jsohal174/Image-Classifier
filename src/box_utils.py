"""
Bounding Box Utility Functions for YOLO Object Detection
Contains functions for IoU calculation and box coordinate conversions
"""

import tensorflow as tf


def iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)

    Returns:
    iou -- Intersection over Union value
    """
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate the Union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area

    return iou


def yolo_boxes_to_corners(box_xy, box_wh):
    """
    Convert YOLO box predictions to bounding box corners.

    Arguments:
    box_xy -- tensor of shape (..., 2) containing box center coordinates
    box_wh -- tensor of shape (..., 2) containing box width and height

    Returns:
    corners -- tensor containing (y_min, x_min, y_max, x_max) coordinates
    """
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


def scale_boxes(boxes, image_shape):
    """
    Scales boxes to match the original image dimensions.

    Arguments:
    boxes -- tensor of shape (None, 4) containing box coordinates
    image_shape -- tuple containing (height, width) of the original image

    Returns:
    scaled_boxes -- boxes scaled to image dimensions
    """
    height = image_shape[0]
    width = image_shape[1]
    image_dims = tf.stack([height, width, height, width])
    image_dims = tf.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes
