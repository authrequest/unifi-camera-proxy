#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

cv2 = importlib.import_module("cv2")


Box = tuple[int, int, int, int]

COCO_PERSON_CLASS_ID = 0
COCO_VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}
DEFAULT_FACE_SCORE_BINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


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


def class_detection_scores(
    detections: list[dict[str, Any]],
    class_name: str | None = None,
) -> list[float]:
    scores: list[float] = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        if class_name is not None and detection.get("class") != class_name:
            continue

        score = detection.get("score")
        if not isinstance(score, (int, float)):
            continue

        numeric = float(score)
        if not math.isfinite(numeric):
            continue
        scores.append(max(0.0, min(1.0, numeric)))

    return scores


def _percentile(sorted_values: list[float], ratio: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    bounded = max(0.0, min(1.0, float(ratio)))
    position = (len(sorted_values) - 1) * bounded
    lower_idx = int(math.floor(position))
    upper_idx = int(math.ceil(position))
    lower = float(sorted_values[lower_idx])
    upper = float(sorted_values[upper_idx])
    if lower_idx == upper_idx:
        return lower
    weight = position - lower_idx
    return (lower * (1.0 - weight)) + (upper * weight)


def build_face_score_telemetry(
    raw_scores: list[float],
    bins: tuple[float, ...] = DEFAULT_FACE_SCORE_BINS,
) -> dict[str, Any]:
    edges = [float(value) for value in bins]
    if len(edges) < 2:
        raise ValueError("face score histogram requires at least two edges")
    if any(not math.isfinite(value) for value in edges):
        raise ValueError("face score histogram edges must be finite")
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError("face score histogram edges must be strictly increasing")

    scores: list[float] = []
    for score in raw_scores:
        if not isinstance(score, (int, float)):
            continue
        numeric = float(score)
        if not math.isfinite(numeric):
            continue
        scores.append(max(edges[0], min(edges[-1], numeric)))

    counts = [0] * (len(edges) - 1)
    for score in scores:
        bucket_index = len(counts) - 1
        for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            in_bucket = (
                left <= score <= right
                if idx == len(counts) - 1
                else left <= score < right
            )
            if in_bucket:
                bucket_index = idx
                break
        counts[bucket_index] += 1

    if not scores:
        summary: dict[str, int | float | None] = {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
        }
    else:
        sorted_scores = sorted(scores)
        summary = {
            "count": len(sorted_scores),
            "min": float(sorted_scores[0]),
            "max": float(sorted_scores[-1]),
            "mean": float(sum(sorted_scores) / len(sorted_scores)),
            "p50": _percentile(sorted_scores, 0.50),
            "p90": _percentile(sorted_scores, 0.90),
        }

    return {
        "summary": summary,
        "histogram": {
            "binEdges": edges,
            "binCounts": counts,
        },
    }


def create_detector(
    backend: str,
    model_path: str | None,
    conf_threshold: float,
    nms_threshold: float,
    input_size: int,
    smartface_param: str | None = None,
    smartface_bin: str | None = None,
    smartface_input_size: int = 112,
    smartface_min_score: float = 0.75,
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

    if normalized == "smartface_ncnn":
        if not smartface_param or not smartface_bin:
            raise ValueError(
                "smartface param and bin paths are required for smartface_ncnn backend"
            )

        param_path = Path(smartface_param)
        bin_path = Path(smartface_bin)
        if not param_path.is_file() or not bin_path.is_file():
            raise ValueError("smartface param/bin path does not exist")

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            raise RuntimeError("unable to initialize OpenCV face cascade")

        ncnn_state: dict[str, Any] = {
            "enabled": False,
            "net": None,
            "ncnn": None,
            "input_blob": None,
            "output_blob": None,
        }

        try:
            ncnn = importlib.import_module("ncnn")
            net = ncnn.Net()
            rc_param = int(net.load_param(str(param_path)))
            rc_model = int(net.load_model(str(bin_path)))
            if rc_param == 0 and rc_model == 0:
                input_names = list(net.input_names())
                output_names = list(net.output_names())
                ncnn_state = {
                    "enabled": bool(input_names and output_names),
                    "net": net,
                    "ncnn": ncnn,
                    "input_blob": input_names[0] if input_names else None,
                    "output_blob": output_names[0] if output_names else None,
                }
        except Exception:
            ncnn_state = {
                "enabled": False,
                "net": None,
                "ncnn": None,
                "input_blob": None,
                "output_blob": None,
            }

        return {
            "name": "smartface_ncnn",
            "face_cascade": cascade,
            "min_score": max(0.0, float(smartface_min_score)),
            "input_size": max(16, int(smartface_input_size)),
            "ncnn": ncnn_state,
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


def infer_face_boxes_from_persons(
    person_boxes: list[Box],
    frame_width: int,
    frame_height: int,
    min_face_size: int = 24,
) -> list[Box]:
    face_boxes: list[Box] = []
    for px, py, pw, ph in person_boxes:
        face_w = max(min_face_size, int(pw * 0.58))
        face_h = max(min_face_size, int(ph * 0.34))
        fx = int(px + (pw - face_w) / 2)
        fy = int(py + ph * 0.04)

        fx = max(0, min(fx, max(0, frame_width - face_w)))
        fy = max(0, min(fy, max(0, frame_height - face_h)))
        if face_w > 0 and face_h > 0 and (face_w * face_h) >= (min_face_size**2):
            face_boxes.append((fx, fy, face_w, face_h))

    return face_boxes


def merge_face_candidate_boxes(
    person_boxes: list[Box],
    motion_boxes: list[Box],
    frame_width: int,
    frame_height: int,
    min_face_size: int = 24,
) -> list[Box]:
    person_candidates = infer_face_boxes_from_persons(
        person_boxes=person_boxes,
        frame_width=frame_width,
        frame_height=frame_height,
        min_face_size=min_face_size,
    )
    if person_candidates:
        return person_candidates

    motion_candidates: list[Box] = []
    for mx, my, mw, mh in motion_boxes:
        face_w = max(min_face_size, int(mw * 0.62))
        face_h = max(min_face_size, int(mh * 0.55))
        fx = int(mx + (mw - face_w) / 2)
        fy = int(my + (mh - face_h) * 0.10)

        fx = max(0, min(fx, max(0, frame_width - face_w)))
        fy = max(0, min(fy, max(0, frame_height - face_h)))
        if face_w > 0 and face_h > 0 and (face_w * face_h) >= (min_face_size**2):
            motion_candidates.append((fx, fy, face_w, face_h))

    deduped: list[Box] = []
    for candidate in motion_candidates:
        if any(rect_iou(candidate, existing) >= 0.55 for existing in deduped):
            continue
        deduped.append(candidate)

    return deduped


def dedupe_face_detections(
    detections: list[dict[str, Any]],
    overlap_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    if len(detections) <= 1:
        return detections

    sorted_detections = sorted(
        detections,
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for current in sorted_detections:
        current_bbox = current.get("bbox")
        if not isinstance(current_bbox, tuple) or len(current_bbox) != 4:
            continue

        should_drop = False
        for previous in kept:
            previous_bbox = previous.get("bbox")
            if not isinstance(previous_bbox, tuple) or len(previous_bbox) != 4:
                continue
            if rect_iou(current_bbox, previous_bbox) >= overlap_threshold:
                should_drop = True
                break

        if not should_drop:
            kept.append(current)

    return kept


def _extract_ncnn_face_confidence(
    detector: dict[str, Any],
    frame_bgr: Any,
    bbox: Box,
) -> float | None:
    ncnn_state = detector.get("ncnn") or {}
    if not ncnn_state.get("enabled"):
        return None

    ncnn = ncnn_state.get("ncnn")
    net = ncnn_state.get("net")
    input_blob = ncnn_state.get("input_blob")
    output_blob = ncnn_state.get("output_blob")
    if not ncnn or not net or not input_blob or not output_blob:
        return None

    x, y, w, h = bbox
    crop = frame_bgr[y : y + h, x : x + w]
    if crop is None or getattr(crop, "size", 0) == 0:
        return None

    input_size = int(detector.get("input_size", 112))

    try:
        resized = cv2.resize(
            crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR
        )
        mat = ncnn.Mat.from_pixels(
            resized,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            int(input_size),
            int(input_size),
        )
        ex = net.create_extractor()
        if int(ex.input(input_blob, mat)) != 0:
            return None
        extract_rc, out_mat = ex.extract(output_blob)
        if int(extract_rc) != 0:
            return None
        out_values = out_mat.numpy("f").reshape(-1)
        if out_values.size == 0:
            return None
        raw = float(abs(out_values).mean())
        norm = raw / (1.0 + raw)
        return max(0.0, min(1.0, norm))
    except Exception:
        return None


def combine_smartface_face_score(
    heuristic_conf: float,
    model_conf: float | None,
) -> float:
    base = float(max(0.0, min(0.99, heuristic_conf)))
    if model_conf is None:
        return base

    model = float(max(0.0, min(1.0, model_conf)))
    boosted_model = 0.55 + (0.45 * model)
    blended = (base * 0.4) + (boosted_model * 0.6)
    return float(max(base, min(0.99, blended)))


def run_smartface_detector(
    detector: dict[str, Any],
    frame_bgr: Any,
    candidate_boxes: list[Box],
) -> list[dict[str, Any]]:
    if detector.get("name") != "smartface_ncnn":
        return []

    cascade = detector["face_cascade"]
    min_score = float(detector.get("min_score", 0.75))
    detections: list[dict[str, Any]] = []

    for cx, cy, cw, ch in candidate_boxes:
        roi = frame_bgr[cy : cy + ch, cx : cx + cw]
        if roi is None or getattr(roi, "size", 0) == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        local_faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(20, 20),
        )

        for lx, ly, lw, lh in local_faces:
            fx, fy, fw, fh = int(cx + lx), int(cy + ly), int(lw), int(lh)
            area_ratio = (fw * fh) / float(
                max(1, frame_bgr.shape[0] * frame_bgr.shape[1])
            )
            heuristic_conf = max(0.55, min(0.92, 0.55 + (area_ratio * 80.0)))
            model_conf = _extract_ncnn_face_confidence(
                detector, frame_bgr, (fx, fy, fw, fh)
            )
            score = combine_smartface_face_score(
                heuristic_conf=heuristic_conf,
                model_conf=model_conf,
            )

            if score < min_score:
                continue

            detections.append(
                {
                    "class": "face",
                    "bbox": (fx, fy, fw, fh),
                    "score": float(max(0.0, min(0.99, score))),
                }
            )

    return dedupe_face_detections(detections, overlap_threshold=0.45)


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


def make_startup_payload(
    rtsp_url: str,
    capture_strategy: str,
    detector_backend: str,
    timestamp_ms: int,
) -> dict[str, Any]:
    normalized = str(rtsp_url).strip().lower()
    secure_rtsp = normalized.startswith("rtsps://")
    rtsp_scheme = "rtsps" if secure_rtsp else "rtsp"
    return {
        "functionName": "ParityStartup",
        "payload": {
            "timestamp": int(timestamp_ms),
            "secureRtsp": bool(secure_rtsp),
            "rtspScheme": rtsp_scheme,
            "captureStrategy": str(capture_strategy),
            "detectorBackend": str(detector_backend),
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


def open_video_capture(rtsp_url: str) -> tuple[Any, str]:
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        return cap, "default"

    is_secure = str(rtsp_url).strip().lower().startswith("rtsps://")
    ffmpeg_flag = getattr(cv2, "CAP_FFMPEG", None)
    if is_secure and ffmpeg_flag is not None:
        fallback_cap = cv2.VideoCapture(rtsp_url, int(ffmpeg_flag))
        if fallback_cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return fallback_cap, "ffmpeg_fallback"

    return cap, "default_failed"


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
    smartface_param = config.get("smartface_param")
    smartface_bin = config.get("smartface_bin")
    smartface_input_size = int(config.get("smartface_input_size", 112))
    smartface_min_score = float(config.get("smartface_min_score", 0.75))
    smartface_stable_frames = max(1, int(config.get("smartface_stable_frames", 2)))

    cap, capture_strategy = open_video_capture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open RTSP stream: {rtsp_url}")

    detector = create_detector(
        backend=detector_backend,
        model_path=(str(detector_model) if detector_model else None),
        conf_threshold=detector_conf_threshold,
        nms_threshold=detector_nms_threshold,
        input_size=detector_input_size,
        smartface_param=(str(smartface_param) if smartface_param else None),
        smartface_bin=(str(smartface_bin) if smartface_bin else None),
        smartface_input_size=smartface_input_size,
        smartface_min_score=smartface_min_score,
    )

    emit_json(
        make_startup_payload(
            rtsp_url=rtsp_url,
            capture_strategy=capture_strategy,
            detector_backend=str(detector.get("name", detector_backend)),
            timestamp_ms=int(time.time() * 1000),
        )
    )

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=24, detectShadows=False
    )
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    prev_classes: set[str] = set()
    event_counter = 0
    face_streak = 0

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
            face_dets: list[dict[str, Any]] = []
            raw_face_count = 0
            stable_face_dets: list[dict[str, Any]] = []

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
            elif detector["name"] == "smartface_ncnn":
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
                face_candidates = merge_face_candidate_boxes(
                    person_boxes=person_boxes,
                    motion_boxes=motion_boxes,
                    frame_width=frame_w,
                    frame_height=frame_h,
                    min_face_size=max(20, int(min(frame_w, frame_h) * 0.035)),
                )
                face_dets = run_smartface_detector(detector, frame, face_candidates)
                raw_face_count = len(face_dets)
                face_streak = face_streak + 1 if face_dets else 0
                stable_face_dets = (
                    face_dets if face_streak >= smartface_stable_frames else []
                )
                all_dets = person_dets + vehicle_dets + stable_face_dets
            else:
                all_dets = run_model_detector(detector, frame)
                person_dets = [d for d in all_dets if d["class"] == "person"]
                vehicle_dets = [d for d in all_dets if d["class"] == "vehicle"]
                face_dets = [d for d in all_dets if d["class"] == "face"]
                raw_face_count = len(face_dets)

            stable_face_count = (
                len(stable_face_dets)
                if detector.get("name") == "smartface_ncnn"
                else raw_face_count
            )
            face_score_telemetry = build_face_score_telemetry(
                class_detection_scores(face_dets, "face")
            )

            curr_classes = {
                str(d.get("class"))
                for d in all_dets
                if isinstance(d, dict) and isinstance(d.get("class"), str)
            }
            enters, leaves = class_edges(prev_classes, curr_classes)
            prev_classes = set(curr_classes)

            now_ms = int(time.time() * 1000)

            for cls in enters:
                event_counter += 1
                confidence_candidates = class_detection_scores(all_dets, cls)
                ev = make_event_payload(
                    edge_type="enter",
                    event_id=f"rtsp-{event_counter}",
                    object_types=[cls],
                    confidence=max(confidence_candidates or [0.5]),
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
                    "faceCount": raw_face_count,
                    "stableFaceCount": stable_face_count,
                    "faceStreak": face_streak,
                    "smartfaceStableFramesThreshold": (
                        smartface_stable_frames
                        if detector.get("name") == "smartface_ncnn"
                        else None
                    ),
                    "faceScoreTelemetry": face_score_telemetry,
                    "motionBoxCount": len(motion_boxes),
                    "detectorBackend": detector.get("name"),
                    "smartfaceRuntime": bool(
                        detector.get("ncnn", {}).get("enabled")
                        if detector.get("name") == "smartface_ncnn"
                        else False
                    ),
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
        help="Detector backend: heuristic, opencv_dnn_yolo, or smartface_ncnn",
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
    parser.add_argument(
        "--smartface-param",
        default=None,
        help="Path to SmartFace NCNN .param file",
    )
    parser.add_argument(
        "--smartface-bin",
        default=None,
        help="Path to SmartFace NCNN .bin file",
    )
    parser.add_argument(
        "--smartface-input-size",
        type=int,
        default=112,
        help="Input size used when probing SmartFace confidence",
    )
    parser.add_argument(
        "--smartface-min-score",
        type=float,
        default=0.75,
        help="Minimum face score to keep SmartFace detections",
    )
    parser.add_argument(
        "--smartface-stable-frames",
        type=int,
        default=2,
        help="Consecutive frames required before emitting face class transitions",
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
        "smartface_param": args.smartface_param,
        "smartface_bin": args.smartface_bin,
        "smartface_input_size": max(16, int(args.smartface_input_size)),
        "smartface_min_score": max(0.0, float(args.smartface_min_score)),
        "smartface_stable_frames": max(1, int(args.smartface_stable_frames)),
    }
    return run_parity(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
