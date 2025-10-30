"""
YOLO Filtering Functions
Contains functions for filtering boxes by score and non-max suppression
"""

import tensorflow as tf


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):
    """
    Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    boxes -- tensor of shape (19, 19, 5, 4)
    box_confidence -- tensor of shape (19, 19, 5, 1)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [highest class probability score < threshold],
                 then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    """
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes using the max box_scores
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1, keepdims=False)

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold"
    filtering_mask = box_class_scores >= threshold

    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask, axis=None)
    boxes = tf.boolean_mask(boxes, filtering_mask, axis=None)
    classes = tf.boolean_mask(box_classes, filtering_mask, axis=None)

    return scores, boxes, classes


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes()
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None,), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    boxes = tf.cast(boxes, dtype=tf.float32)
    scores = tf.cast(scores, dtype=tf.float32)

    nms_indices = []
    classes_labels = tf.unique(classes)[0]  # Get unique classes

    for label in classes_labels:
        filtering_mask = classes == label

        # Get boxes for this class
        boxes_label = tf.boolean_mask(boxes, filtering_mask)

        # Get scores for this class
        scores_label = tf.boolean_mask(scores, filtering_mask)

        if tf.shape(scores_label)[0] > 0:  # Check if there are any boxes to process
            # Use tf.image.non_max_suppression()
            nms_indices_label = tf.image.non_max_suppression(
                boxes_label,
                scores_label,
                max_boxes,
                iou_threshold=iou_threshold
            )

            # Get original indices of the selected boxes
            selected_indices = tf.squeeze(tf.where(filtering_mask), axis=1)

            # Append the resulting boxes into the partial result
            nms_indices.append(tf.gather(selected_indices, nms_indices_label))

    # Flatten the list of indices and concatenate
    nms_indices = tf.concat(nms_indices, axis=0)

    # Use tf.gather() to select only nms_indices from scores, boxes and classes
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)

    # Sort by scores and return the top max_boxes
    sort_order = tf.argsort(scores, direction='DESCENDING').numpy()
    scores = tf.gather(scores, sort_order[0:max_boxes])
    boxes = tf.gather(boxes, sort_order[0:max_boxes])
    classes = tf.gather(classes, sort_order[0:max_boxes])

    return scores, boxes, classes
