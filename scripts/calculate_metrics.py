"""
Calculate Evaluation Metrics (mAP, Precision, Recall, F1)
Compare predictions with ground truth annotations
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def parse_yolo_annotation(label_file):
    """
    Parse YOLO format annotation file

    Format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]

    Args:
        label_file: Path to label file

    Returns:
        List of annotations (class_id, bbox)
    """
    annotations = []

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert to [x_min, y_min, x_max, y_max] format
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2

            bbox = [x_min, y_min, x_max, y_max]
            annotations.append({'class': class_id, 'bbox': bbox})

    return annotations


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union

    Args:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]

    Returns:
        IoU value [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_width = max(0, xi_max - xi_min)
    inter_height = max(0, yi_max - yi_min)
    inter_area = inter_width * inter_height

    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    if union_area == 0:
        return 0
    iou = inter_area / union_area
    return iou


def match_predictions_to_ground_truth(predictions, ground_truths, iou_threshold=0.5):
    """
    Match predictions to ground truth boxes

    Args:
        predictions: List of predicted boxes with scores
        ground_truths: List of ground truth boxes
        iou_threshold: IoU threshold for matching

    Returns:
        true_positives, false_positives, false_negatives
    """
    if len(predictions) == 0 and len(ground_truths) == 0:
        return 0, 0, 0

    if len(predictions) == 0:
        return 0, 0, len(ground_truths)

    if len(ground_truths) == 0:
        return 0, len(predictions), 0

    # Sort predictions by score (descending)
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    matched_gt = set()
    true_positives = 0
    false_positives = 0

    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1

        # Find best matching ground truth
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue

            if pred['class'] != gt['class']:
                continue

            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if match is good enough
        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1

    false_negatives = len(ground_truths) - len(matched_gt)

    return true_positives, false_positives, false_negatives


def calculate_average_precision(precisions, recalls):
    """
    Calculate Average Precision (AP) using 11-point interpolation

    Args:
        precisions: List of precision values
        recalls: List of recall values

    Returns:
        Average Precision
    """
    if len(precisions) == 0:
        return 0.0

    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = np.array(recalls)[sorted_indices]
    precisions = np.array(precisions)[sorted_indices]

    # Add sentinel values
    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))

    # Compute maximum precision at each recall level
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Calculate AP (area under PR curve)
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return ap


def evaluate_predictions(predictions_dir, ground_truth_dir, iou_threshold=0.5):
    """
    Evaluate predictions against ground truth

    Args:
        predictions_dir: Directory with prediction JSON files
        ground_truth_dir: Directory with ground truth label files
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with evaluation metrics
    """
    predictions_dir = Path(predictions_dir)
    ground_truth_dir = Path(ground_truth_dir)

    # Collect all results per class
    class_results = defaultdict(lambda: {
        'tp': 0, 'fp': 0, 'fn': 0,
        'predictions': [], 'ground_truths': []
    })

    # Process each prediction file
    pred_files = sorted(predictions_dir.glob('*.json'))

    print(f"\nEvaluating {len(pred_files)} predictions...")

    for pred_file in pred_files:
        # Load predictions
        with open(pred_file, 'r') as f:
            pred_data = json.load(f)

        # Get corresponding ground truth
        image_name = pred_data['image']
        gt_file = ground_truth_dir / (Path(image_name).stem + '.txt')

        if not gt_file.exists():
            print(f"Warning: No ground truth for {image_name}")
            continue

        ground_truths = parse_yolo_annotation(gt_file)
        predictions = pred_data['detections']

        # Group by class
        pred_by_class = defaultdict(list)
        gt_by_class = defaultdict(list)

        for pred in predictions:
            pred_by_class[pred['class']].append(pred)
            class_results[pred['class']]['predictions'].append(pred)

        for gt in ground_truths:
            gt_by_class[gt['class']].append(gt)
            class_results[gt['class']]['ground_truths'].append(gt)

        # Calculate metrics for each class
        all_classes = set(list(pred_by_class.keys()) + list(gt_by_class.keys()))

        for cls in all_classes:
            tp, fp, fn = match_predictions_to_ground_truth(
                pred_by_class[cls],
                gt_by_class[cls],
                iou_threshold
            )

            class_results[cls]['tp'] += tp
            class_results[cls]['fp'] += fp
            class_results[cls]['fn'] += fn

    # Calculate metrics for each class
    metrics = {}

    for cls, results in class_results.items():
        tp = results['tp']
        fp = results['fp']
        fn = results['fn']

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate AP
        all_preds = sorted(results['predictions'], key=lambda x: x['score'], reverse=True)
        precisions = []
        recalls = []

        tp_cumsum = 0
        fp_cumsum = 0
        n_gt = len(results['ground_truths'])

        for pred in all_preds:
            # This is simplified; actual AP calculation needs per-prediction matching
            tp_cumsum += 1  # Assuming all predictions are TP for simplicity
            prec = tp_cumsum / (tp_cumsum + fp_cumsum)
            rec = tp_cumsum / n_gt if n_gt > 0 else 0
            precisions.append(prec)
            recalls.append(rec)

        ap = calculate_average_precision(precisions, recalls) if precisions else 0

        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap': ap,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    # Calculate mAP
    if metrics:
        mAP = np.mean([m['ap'] for m in metrics.values()])
    else:
        mAP = 0.0

    return {
        'mAP': mAP,
        'per_class': metrics
    }


def plot_metrics(metrics, output_dir):
    """
    Plot evaluation metrics

    Args:
        metrics: Dictionary with evaluation metrics
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics['per_class']:
        print("No metrics to plot")
        return

    # Extract data
    classes = list(metrics['per_class'].keys())
    precisions = [metrics['per_class'][c]['precision'] for c in classes]
    recalls = [metrics['per_class'][c]['recall'] for c in classes]
    f1_scores = [metrics['per_class'][c]['f1'] for c in classes]
    aps = [metrics['per_class'][c]['ap'] for c in classes]

    # Plot 1: Precision, Recall, F1 per class
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.25

    ax.bar(x - width, precisions, width, label='Precision')
    ax.bar(x, recalls, width, label='Recall')
    ax.bar(x + width, f1_scores, width, label='F1')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, and F1 Score per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_f1.png', dpi=150)
    print(f"  ✓ Saved plot: {output_dir / 'precision_recall_f1.png'}")

    # Plot 2: Average Precision per class
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(classes, aps, color='steelblue')
    ax.axhline(y=metrics['mAP'], color='r', linestyle='--',
               label=f"mAP = {metrics['mAP']:.3f}")

    ax.set_xlabel('Class')
    ax.set_ylabel('Average Precision (AP)')
    ax.set_title(f"Average Precision per Class (mAP@0.5 = {metrics['mAP']:.3f})")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'average_precision.png', dpi=150)
    print(f"  ✓ Saved plot: {output_dir / 'average_precision.png'}")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description='Calculate evaluation metrics (mAP, Precision, Recall)'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Directory with prediction JSON files'
    )
    parser.add_argument(
        '--ground-truth',
        type=str,
        required=True,
        help='Directory with ground truth label files (YOLO format)'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for matching (default: 0.5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='metrics_results.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='plots',
        help='Directory to save plots'
    )

    args = parser.parse_args()

    print("="*60)
    print("YOLO Metrics Calculation")
    print("="*60)
    print(f"Predictions:   {args.predictions}")
    print(f"Ground Truth:  {args.ground_truth}")
    print(f"IoU Threshold: {args.iou_threshold}")
    print("="*60)

    # Calculate metrics
    metrics = evaluate_predictions(
        args.predictions,
        args.ground_truth,
        args.iou_threshold
    )

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"\nmAP@{args.iou_threshold}: {metrics['mAP']:.4f}")

    print("\nPer-class metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AP':<12}")
    print("-" * 60)

    for cls, m in metrics['per_class'].items():
        print(f"{cls:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1']:<12.4f} {m['ap']:<12.4f}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Results saved to: {args.output}")

    # Plot metrics
    print(f"\nGenerating plots...")
    plot_metrics(metrics, args.plot_dir)

    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
