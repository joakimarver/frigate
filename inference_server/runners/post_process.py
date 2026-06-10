"""Post-processing functions that convert raw model outputs to Frigate's
(20, 6) float32 detection format.

Each function returns a numpy array of shape (20, 6) where each row is:
    [class_id, confidence, y_min_norm, x_min_norm, y_max_norm, x_max_norm]
with coordinates normalized to [0, 1] relative to the model input dimensions.

These functions are ported verbatim from ``frigate/util/model.py`` so that
the remote inference server produces identical results to the embedded Frigate
detectors.
"""

import cv2
import numpy as np

# Confidence threshold used consistently across all post-processors.
_CONF_THRESHOLD = 0.4
_NMS_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# DFINE
# ---------------------------------------------------------------------------


def post_process_dfine(
    tensor_output: list[np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    class_ids = tensor_output[0][tensor_output[2] > _CONF_THRESHOLD]
    boxes = tensor_output[1][tensor_output[2] > _CONF_THRESHOLD]
    scores = tensor_output[2][tensor_output[2] > _CONF_THRESHOLD]

    input_shape = np.array([height, width, height, width])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(
        boxes, scores, score_threshold=_CONF_THRESHOLD, nms_threshold=_NMS_THRESHOLD
    )
    detections = np.zeros((20, 6), np.float32)

    for i, (bbox, confidence, class_id) in enumerate(
        zip(boxes[indices], scores[indices], class_ids[indices])
    ):
        if i == 20:
            break
        detections[i] = [
            class_id,
            confidence,
            bbox[1],
            bbox[0],
            bbox[3],
            bbox[2],
        ]

    return detections


# ---------------------------------------------------------------------------
# RF-DETR
# ---------------------------------------------------------------------------


def post_process_rfdetr(tensor_output: list[np.ndarray]) -> np.ndarray:
    boxes = tensor_output[0]
    raw_scores = tensor_output[1]

    exp = np.exp(raw_scores - np.max(raw_scores, axis=-1, keepdims=True))
    all_scores = exp / np.sum(exp, axis=-1, keepdims=True)

    scores = np.max(all_scores[0, :, 1:], axis=-1)
    labels = np.argmax(all_scores[0, :, 1:], axis=-1)

    idxs = scores > _CONF_THRESHOLD
    filtered_boxes = boxes[0, idxs]
    filtered_scores = scores[idxs]
    filtered_labels = labels[idxs]

    x_center, y_center, w, h = (
        filtered_boxes[:, 0],
        filtered_boxes[:, 1],
        filtered_boxes[:, 2],
        filtered_boxes[:, 3],
    )
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2
    filtered_boxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)

    indices = cv2.dnn.NMSBoxes(
        filtered_boxes,
        filtered_scores,
        score_threshold=_CONF_THRESHOLD,
        nms_threshold=_NMS_THRESHOLD,
    )
    detections = np.zeros((20, 6), np.float32)

    for i, (bbox, confidence, class_id) in enumerate(
        zip(
            filtered_boxes[indices],
            filtered_scores[indices],
            filtered_labels[indices],
        )
    ):
        if i == 20:
            break
        detections[i] = [
            class_id,
            confidence,
            bbox[1],
            bbox[0],
            bbox[3],
            bbox[2],
        ]

    return detections


# ---------------------------------------------------------------------------
# YOLO-X
# ---------------------------------------------------------------------------


def post_process_yolox(
    predictions: np.ndarray,
    width: int,
    height: int,
    grids: np.ndarray,
    expanded_strides: np.ndarray,
) -> np.ndarray:
    predictions[..., :2] = (predictions[..., :2] + grids) * expanded_strides
    predictions[..., 2:4] = np.exp(predictions[..., 2:4]) * expanded_strides

    predictions = predictions[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    cls_inds = scores.argmax(1)
    scores = scores[np.arange(len(cls_inds)), cls_inds]

    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy,
        scores,
        score_threshold=_CONF_THRESHOLD,
        nms_threshold=_NMS_THRESHOLD,
    )

    detections = np.zeros((20, 6), np.float32)
    for i, (bbox, confidence, class_id) in enumerate(
        zip(boxes_xyxy[indices], scores[indices], cls_inds[indices])
    ):
        if i == 20:
            break
        detections[i] = [
            class_id,
            confidence,
            bbox[1] / height,
            bbox[0] / width,
            bbox[3] / height,
            bbox[2] / width,
        ]

    return detections


# ---------------------------------------------------------------------------
# YOLO-Generic (single-output NMS variant and multi-output anchor variant)
# ---------------------------------------------------------------------------


def _post_process_multipart_yolo(
    output_list: list[np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    anchors_per_scale = [
        np.array([(12, 16), (19, 36), (40, 28)], dtype=np.float32),
        np.array([(36, 75), (76, 55), (72, 146)], dtype=np.float32),
        np.array([(142, 110), (192, 243), (459, 401)], dtype=np.float32),
    ]
    stride_map = {0: 8, 1: 16, 2: 32}

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_class_ids: list[np.ndarray] = []

    for i, output in enumerate(output_list):
        bs, _, ny, nx = output.shape
        stride = stride_map[i]
        anchor_set = anchors_per_scale[i]  # (3, 2)
        num_anchors = len(anchor_set)

        # Reshape to (num_anchors, ny, nx, 85)
        output = output.reshape(bs, num_anchors, 85, ny, nx)
        output = output.transpose(0, 1, 3, 4, 2)[0]  # (num_anchors, ny, nx, 85)

        # Build grid coordinates — shape (ny, nx, 2)
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        grid = np.stack([xv, yv], axis=-1).astype(np.float32)  # (ny, nx, 2)

        # Decode box centres: (anchor, ny, nx)
        pred_xy = output[..., :2]  # (A, ny, nx, 2)
        pred_wh = output[..., 2:4]
        objectness = output[..., 4]  # (A, ny, nx)
        class_probs = output[..., 5:]  # (A, ny, nx, num_classes)

        bx = ((pred_xy[..., 0] * 2.0 - 0.5) + grid[:, :, 0]) * stride
        by = ((pred_xy[..., 1] * 2.0 - 0.5) + grid[:, :, 1]) * stride

        # anchor_set[:, 0] -> (A,) broadcast to (A, ny, nx)
        bw = ((pred_wh[..., 0] * 2.0) ** 2) * anchor_set[:, 0, np.newaxis, np.newaxis]
        bh = ((pred_wh[..., 1] * 2.0) ** 2) * anchor_set[:, 1, np.newaxis, np.newaxis]

        class_conf = np.max(class_probs, axis=-1)  # (A, ny, nx)
        class_id = np.argmax(class_probs, axis=-1)  # (A, ny, nx)
        conf = class_conf * objectness  # (A, ny, nx)

        # Filter by confidence
        mask = conf >= _CONF_THRESHOLD
        if not np.any(mask):
            continue

        x1 = np.clip(bx[mask] - bw[mask] / 2, 0.0, float(width))
        y1 = np.clip(by[mask] - bh[mask] / 2, 0.0, float(height))
        x2 = np.clip(bx[mask] + bw[mask] / 2, 0.0, float(width))
        y2 = np.clip(by[mask] + bh[mask] / 2, 0.0, float(height))

        all_boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
        all_scores.append(conf[mask])
        all_class_ids.append(class_id[mask])

    if not all_boxes:
        return np.zeros((20, 6), np.float32)

    all_boxes_np = np.concatenate(all_boxes, axis=0).tolist()
    all_scores_np = np.concatenate(all_scores, axis=0).tolist()
    all_class_ids_np = np.concatenate(all_class_ids, axis=0)

    indices = cv2.dnn.NMSBoxes(
        bboxes=all_boxes_np,
        scores=all_scores_np,
        score_threshold=_CONF_THRESHOLD,
        nms_threshold=_NMS_THRESHOLD,
    )

    results = np.zeros((20, 6), np.float32)
    if len(indices) > 0:
        for out_idx, idx in enumerate(indices.flatten()[:20]):
            class_id = int(all_class_ids_np[idx])
            conf = float(all_scores_np[idx])
            x1, y1, x2, y2 = all_boxes_np[idx]
            results[out_idx] = [
                class_id,
                conf,
                y1 / height,
                x1 / width,
                y2 / height,
                x2 / width,
            ]

    return results


def _post_process_nms_yolo(
    predictions: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    predictions = np.squeeze(predictions)

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > _CONF_THRESHOLD, :]
    scores = scores[scores > _CONF_THRESHOLD]
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    boxes = predictions[:, :4]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    boxes = boxes_xyxy

    indices = cv2.dnn.NMSBoxes(
        boxes, scores, score_threshold=_CONF_THRESHOLD, nms_threshold=_NMS_THRESHOLD
    )
    detections = np.zeros((20, 6), np.float32)
    for i, (bbox, confidence, class_id) in enumerate(
        zip(boxes[indices], scores[indices], class_ids[indices])
    ):
        if i == 20:
            break
        detections[i] = [
            class_id,
            confidence,
            bbox[1] / height,
            bbox[0] / width,
            bbox[3] / height,
            bbox[2] / width,
        ]

    return detections


def post_process_yolo(
    output: list[np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    if len(output) > 1:
        return _post_process_multipart_yolo(output, width, height)
    return _post_process_nms_yolo(output[0], width, height)
