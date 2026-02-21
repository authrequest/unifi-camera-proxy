#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import cv2


Box = tuple[int, int, int, int]

COCO_PERSON_CLASS_ID = 0
COCO_VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}


def rect_iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def class_edges(
    prev_classes: set[str], curr_classes: set[str]
) -> tuple[list[str], list[str]]:
    enters = sorted(curr_classes - prev_classes)
    leaves = sorted(prev_classes - curr_classes)
    return enters, leaves


def coco_id_to_token(class_id: int) -> str | None:
    if class_id == COCO_PERSON_CLASS_ID:
        return "person"
    if class_id in COCO_VEHICLE_CLASS_IDS:
        return "vehicle"
    return None


def create_detector(
    backend: str,
    model_path: str | None,
    conf_threshold: float,
    nms_threshold: float,
    input_size: int,
) -> dict[str, Any]:
    normalized = str(backend or "heuristic").strip().lower()
    if normalized == "heuristic":
        return {"name": "heuristic"}

    if normalized in {"opencv_dnn_yolo", "yolo"}:
        if not model_path:
            raise ValueError(
                "detector model path is required for opencv_dnn_yolo backend"
            )

        net = cv2.dnn.readNet(model_path)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(
            size=(int(input_size), int(input_size)),
            scale=1.0 / 255.0,
            swapRB=True,
        )

        return {
            "name": "opencv_dnn_yolo",
            "model": model,
            "conf_threshold": float(conf_threshold),
            "nms_threshold": float(nms_threshold),
        }

    raise ValueError(f"unsupported detector backend: {backend}")


def run_model_detector(
    detector: dict[str, Any], frame_bgr: Any
) -> list[dict[str, Any]]:
    if detector.get("name") != "opencv_dnn_yolo":
        return []

    model = detector["model"]
    conf_threshold = float(detector["conf_threshold"])
    nms_threshold = float(detector["nms_threshold"])

    class_ids, confidences, boxes = model.detect(
        frame_bgr,
        confThreshold=conf_threshold,
        nmsThreshold=nms_threshold,
    )

    detections: list[dict[str, Any]] = []
    if class_ids is None or confidences is None or boxes is None:
        return detections

    class_ids_flat = [int(x) for x in list(class_ids.reshape(-1))]
    confidences_flat = [float(x) for x in list(confidences.reshape(-1))]
    boxes_flat = [tuple(map(int, b)) for b in list(boxes.reshape(-1, 4))]

    for class_id, score, box in zip(class_ids_flat, confidences_flat, boxes_flat):
        token = coco_id_to_token(class_id)
        if token is None and class_id > 0:
            token = coco_id_to_token(class_id - 1)
        if token is None:
            continue

        detections.append(
            {
                "class": token,
                "bbox": box,
                "score": float(score),
            }
        )

    return detections


def infer_vehicle_from_motion(
    motion_boxes: list[Box],
    person_boxes: list[Box],
    frame_area: int,
    min_area_ratio: float = 0.004,
    overlap_iou_drop: float = 0.35,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    min_area = max(1, int(frame_area * min_area_ratio))

    for mx, my, mw, mh in motion_boxes:
        area = mw * mh
        if area < min_area:
            continue

        overlaps_person = False
        for pbox in person_boxes:
            if rect_iou((mx, my, mw, mh), pbox) >= overlap_iou_drop:
                overlaps_person = True
                break

        if overlaps_person:
            continue

        conf = min(0.99, max(0.5, area / float(max(frame_area, 1))))
        detections.append(
            {
                "class": "vehicle",
                "bbox": (mx, my, mw, mh),
                "score": float(conf),
            }
        )

    return detections


def make_event_payload(
    edge_type: str,
    event_id: str,
    object_types: list[str],
    confidence: float,
    timestamp_ms: int,
) -> dict[str, Any]:
    return {
        "functionName": "EventSmartDetect",
        "payload": {
            "edgeType": edge_type,
            "eventId": event_id,
            "eventType": "motion",
            "timestamp": timestamp_ms,
            "objectTypes": object_types,
            "zonesStatus": [
                {
                    "zoneIds": [1],
                    "objectTypes": object_types,
                    "confidence": round(float(confidence), 4),
                }
            ],
        },
    }


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _detect_motion_boxes(fg_mask: Any, min_area: int) -> list[Box]:
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def _detect_person_boxes(hog: Any, frame_bgr: Any) -> list[Box]:
    rects, _weights = hog.detectMultiScale(
        frame_bgr,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    boxes: list[Box] = []
    for x, y, w, h in rects:
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def run_parity(config: dict[str, Any]) -> int:
    rtsp_url = str(config["rtsp_url"])
    max_frames = int(config["max_frames"])
    motion_area_ratio_min = float(config["motion_area_ratio_min"])
    sleep_ms = int(config["sleep_ms"])
    detector_backend = str(config.get("detector_backend", "heuristic"))
    detector_model = config.get("detector_model")
    detector_conf_threshold = float(config.get("detector_conf_threshold", 0.35))
    detector_nms_threshold = float(config.get("detector_nms_threshold", 0.45))
    detector_input_size = int(config.get("detector_input_size", 640))

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open RTSP stream: {rtsp_url}")

    detector = create_detector(
        backend=detector_backend,
        model_path=(str(detector_model) if detector_model else None),
        conf_threshold=detector_conf_threshold,
        nms_threshold=detector_nms_threshold,
        input_size=detector_input_size,
    )

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=24, detectShadows=False
    )
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    prev_classes: set[str] = set()
    event_counter = 0

    try:
        idx = 0
        while True:
            if max_frames > 0 and idx >= max_frames:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_h, frame_w = frame.shape[:2]
            frame_area = max(1, frame_h * frame_w)
            min_motion_area = max(1, int(frame_area * motion_area_ratio_min))

            fg = subtractor.apply(frame)
            fg = cv2.medianBlur(fg, 5)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

            motion_boxes = _detect_motion_boxes(fg, min_motion_area)

            if detector["name"] == "heuristic":
                person_boxes = _detect_person_boxes(hog, frame)
                person_dets = [
                    {"class": "person", "bbox": b, "score": 0.75} for b in person_boxes
                ]
                vehicle_dets = infer_vehicle_from_motion(
                    motion_boxes=motion_boxes,
                    person_boxes=person_boxes,
                    frame_area=frame_area,
                    min_area_ratio=motion_area_ratio_min,
                )
                all_dets = person_dets + vehicle_dets
            else:
                all_dets = run_model_detector(detector, frame)
                person_dets = [d for d in all_dets if d["class"] == "person"]
                vehicle_dets = [d for d in all_dets if d["class"] == "vehicle"]
            curr_classes = {d["class"] for d in all_dets}
            enters, leaves = class_edges(prev_classes, curr_classes)
            prev_classes = set(curr_classes)

            now_ms = int(time.time() * 1000)

            for cls in enters:
                event_counter += 1
                ev = make_event_payload(
                    edge_type="enter",
                    event_id=f"rtsp-{event_counter}",
                    object_types=[cls],
                    confidence=max(
                        [float(d["score"]) for d in all_dets if d["class"] == cls]
                        or [0.5]
                    ),
                    timestamp_ms=now_ms,
                )
                emit_json(ev)

            for cls in leaves:
                event_counter += 1
                ev = make_event_payload(
                    edge_type="leave",
                    event_id=f"rtsp-{event_counter}",
                    object_types=[cls],
                    confidence=0.5,
                    timestamp_ms=now_ms,
                )
                emit_json(ev)

            heartbeat = {
                "functionName": "ParityFrameSummary",
                "payload": {
                    "frameIndex": idx,
                    "timestamp": now_ms,
                    "classes": sorted(curr_classes),
                    "personCount": len(person_dets),
                    "vehicleCount": len(vehicle_dets),
                    "motionBoxCount": len(motion_boxes),
                },
            }
            emit_json(heartbeat)

            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

            idx += 1
    finally:
        cap.release()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RTSP parity detector and emit UniFi-like SmartDetect events"
    )
    parser.add_argument("--rtsp-url", required=True, help="RTSP URL")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process (0 = unlimited)",
    )
    parser.add_argument(
        "--motion-area-ratio-min",
        type=float,
        default=0.004,
        help="Minimum foreground area ratio to consider a motion box",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=0,
        help="Optional sleep per frame in milliseconds",
    )
    parser.add_argument(
        "--detector-backend",
        default="heuristic",
        help="Detector backend: heuristic or opencv_dnn_yolo",
    )
    parser.add_argument(
        "--detector-model",
        default=None,
        help="Path to detector model when backend requires one",
    )
    parser.add_argument(
        "--detector-conf-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for model detector backends",
    )
    parser.add_argument(
        "--detector-nms-threshold",
        type=float,
        default=0.45,
        help="NMS threshold for model detector backends",
    )
    parser.add_argument(
        "--detector-input-size",
        type=int,
        default=640,
        help="Square input size for model detector backends",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = {
        "rtsp_url": args.rtsp_url,
        "max_frames": max(0, int(args.max_frames)),
        "motion_area_ratio_min": max(1e-6, float(args.motion_area_ratio_min)),
        "sleep_ms": max(0, int(args.sleep_ms)),
        "detector_backend": str(args.detector_backend),
        "detector_model": args.detector_model,
        "detector_conf_threshold": max(0.0, float(args.detector_conf_threshold)),
        "detector_nms_threshold": max(0.0, float(args.detector_nms_threshold)),
        "detector_input_size": max(1, int(args.detector_input_size)),
    }
    return run_parity(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
