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
ENROLL_REQUIRED_OUTPUTS = (
    "regression_stride_16",
    "regression_stride_32",
    "regression_stride_64",
    "classification_stride_16",
    "classification_stride_32",
    "classification_stride_64",
)
ENROLL_ANCHOR_SCALE_HINTS: dict[int, tuple[float, ...]] = {
    16: (1.05, 1.45),
    32: (1.15, 1.65),
    64: (0.80, 1.00, 1.20, 1.35, 1.50, 1.65),
}
ENROLL_VARIANCE_CENTER = 0.1
ENROLL_VARIANCE_SIZE = 0.2
ENROLL_CANDIDATE_SCORE_FLOOR = 0.2
SMARTFACE_LMK_PARAM_CANDIDATES = (
    "SmartFace_lmk_20220304.param",
    "SmartFace_lmk.param",
)
SMARTFACE_LMK_BIN_CANDIDATES = (
    "SmartFace_lmk_20220304.bin",
    "SmartFace_lmk.bin",
)
SMARTFACE_EXTRACT_PARAM_CANDIDATES = (
    "SmartFace_extract_20220317.param",
    "SmartFace_extract.param",
)
SMARTFACE_EXTRACT_BIN_CANDIDATES = (
    "SmartFace_extract_20220317.bin",
    "SmartFace_extract.bin",
)
DEFAULT_IDENTITY_GALLERY_MAX_PROFILES = 2000
DEFAULT_IDENTITY_GALLERY_SAVE_INTERVAL_FRAMES = 30
DEFAULT_IDENTITY_GALLERY_MAX_IDLE_MS = 1000 * 60 * 60 * 24 * 30
DEFAULT_IDENTITY_GALLERY_PRUNE_INTERVAL_FRAMES = 30
DEFAULT_IDENTITY_SPLIT_GUARD_RATIO = 0.9
DEFAULT_IDENTITY_SPLIT_GUARD_MAX_SEEN = 3
DEFAULT_IDENTITY_MERGE_RECOVER_THRESHOLD = 0.45
DEFAULT_IDENTITY_MERGE_RECOVER_MIN_SEEN = 5
DEFAULT_IDENTITY_QUALITY_VERIFIER_ENABLED = True
DEFAULT_IDENTITY_POSE_MASK_GATE_ENABLED = False
DEFAULT_IDENTITY_MAX_ABS_YAW_DEG = 45.0
DEFAULT_IDENTITY_MAX_ABS_PITCH_DEG = 35.0
DEFAULT_IDENTITY_MAX_ABS_ROLL_DEG = 30.0
DEFAULT_IDENTITY_MASK_CONFIDENCE_MIN = 0.5
IDENTITY_AUDIT_DECISIONS = (
    "strict_match",
    "merge_recover",
    "new_identity_split_guard",
    "new_identity_reserved",
    "new_identity_threshold",
    "new_identity_no_profiles",
    "new_identity_invalid_embedding",
    "unknown",
)
IDENTITY_QUALITY_REASONS = (
    "quality_ok",
    "quality_ok_telemetry_only",
    "quality_verifier_disabled",
    "pose_unavailable",
    "mask_unavailable",
    "pose_out_of_range",
    "mask_confidence_low",
    "unknown",
)


def new_identity_decision_counters() -> dict[str, int]:
    return {str(code): 0 for code in IDENTITY_AUDIT_DECISIONS}


def bump_identity_decision_counter(
    counters: dict[str, int],
    decision: str | None,
) -> None:
    code = (
        str(decision)
        if isinstance(decision, str) and decision in counters
        else "unknown"
    )
    counters[code] = int(counters.get(code, 0)) + 1


def new_identity_quality_counters() -> dict[str, int]:
    return {str(code): 0 for code in IDENTITY_QUALITY_REASONS}


def bump_identity_quality_counter(
    counters: dict[str, int],
    reason: str | None,
) -> str:
    code = str(reason) if isinstance(reason, str) and reason in counters else "unknown"
    counters[code] = int(counters.get(code, 0)) + 1
    return code


def _coerce_pose_angles(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, tuple) and len(value) == 3:
        if all(
            isinstance(component, (int, float)) and math.isfinite(float(component))
            for component in value
        ):
            return (float(value[0]), float(value[1]), float(value[2]))
        return None
    if isinstance(value, list) and len(value) == 3:
        if all(
            isinstance(component, (int, float)) and math.isfinite(float(component))
            for component in value
        ):
            return (float(value[0]), float(value[1]), float(value[2]))
        return None
    if isinstance(value, dict):
        yaw = value.get("yaw")
        pitch = value.get("pitch")
        roll = value.get("roll")
        if (
            isinstance(yaw, (int, float))
            and math.isfinite(float(yaw))
            and isinstance(pitch, (int, float))
            and math.isfinite(float(pitch))
            and isinstance(roll, (int, float))
            and math.isfinite(float(roll))
        ):
            return (float(yaw), float(pitch), float(roll))
    return None


def _coerce_mask_confidence(face_detection: dict[str, Any]) -> float | None:
    value = face_detection.get("maskConfidence")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def apply_pose_mask_quality_verifier(
    face_detection: dict[str, Any],
    verifier_enabled: bool,
    hard_gate_enabled: bool,
    max_abs_yaw: float,
    max_abs_pitch: float,
    max_abs_roll: float,
    mask_confidence_min: float,
    quality_stats_frame: dict[str, int],
    quality_stats_cumulative: dict[str, int],
    identity_reject_reasons: dict[str, int],
) -> tuple[bool, str, dict[str, Any]]:
    accepted, reason, details = evaluate_pose_mask_quality(
        pose_angles=_coerce_pose_angles(face_detection.get("poseAngles")),
        mask_confidence=_coerce_mask_confidence(face_detection),
        verifier_enabled=bool(verifier_enabled),
        hard_gate_enabled=bool(hard_gate_enabled),
        max_abs_yaw=float(max_abs_yaw),
        max_abs_pitch=float(max_abs_pitch),
        max_abs_roll=float(max_abs_roll),
        mask_confidence_min=float(mask_confidence_min),
    )
    reason_code = bump_identity_quality_counter(quality_stats_frame, reason)
    bump_identity_quality_counter(quality_stats_cumulative, reason)
    if not accepted:
        identity_reject_reasons[reason_code] = (
            int(identity_reject_reasons.get(reason_code, 0)) + 1
        )
    return accepted, reason_code, details


def build_identity_quality_summary_payload(
    verifier_enabled: bool,
    hard_gate_enabled: bool,
    quality_stats_frame: dict[str, int],
    quality_stats_cumulative: dict[str, int],
    max_abs_yaw: float,
    max_abs_pitch: float,
    max_abs_roll: float,
    mask_confidence_min: float,
) -> dict[str, Any]:
    return {
        "identityQualityVerifierEnabled": bool(verifier_enabled),
        "identityPoseMaskGateEnabled": bool(hard_gate_enabled),
        "identityQualityStatsFrame": dict(quality_stats_frame),
        "identityQualityStatsCumulative": dict(quality_stats_cumulative),
        "identityMaxAbsYawDeg": float(max_abs_yaw),
        "identityMaxAbsPitchDeg": float(max_abs_pitch),
        "identityMaxAbsRollDeg": float(max_abs_roll),
        "identityMaskConfidenceMin": float(mask_confidence_min),
    }


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


def _resolve_model_pair(
    explicit_param: str | None,
    explicit_bin: str | None,
    base_dir: Path,
    param_candidates: tuple[str, ...],
    bin_candidates: tuple[str, ...],
) -> tuple[Path, Path] | None:
    if explicit_param and explicit_bin:
        param_path = Path(explicit_param).resolve()
        bin_path = Path(explicit_bin).resolve()
        if param_path.is_file() and bin_path.is_file():
            return param_path, bin_path
        return None

    for param_name, bin_name in zip(param_candidates, bin_candidates):
        param_path = (base_dir / param_name).resolve()
        bin_path = (base_dir / bin_name).resolve()
        if param_path.is_file() and bin_path.is_file():
            return param_path, bin_path

    return None


def _load_optional_ncnn_model(
    ncnn: Any,
    model_pair: tuple[Path, Path] | None,
    *,
    input_size: int,
    pixel_type: str,
    mean_vals: list[float] | None,
    norm_vals: list[float] | None,
    preferred_output: str | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "enabled": False,
        "ncnn": ncnn,
        "net": None,
        "input_blob": None,
        "output_blob": None,
        "output_heads": tuple(),
        "input_size": max(16, int(input_size)),
        "pixel_type": str(pixel_type),
        "mean_vals": mean_vals,
        "norm_vals": norm_vals,
    }
    if model_pair is None:
        return state

    param_path, bin_path = model_pair
    try:
        net = ncnn.Net()
        rc_param = int(net.load_param(str(param_path)))
        rc_model = int(net.load_model(str(bin_path)))
        if rc_param != 0 or rc_model != 0:
            return state

        input_names = tuple(str(name) for name in list(net.input_names()))
        output_names = tuple(str(name) for name in list(net.output_names()))
        if not input_names or not output_names:
            return state

        output_blob = (
            preferred_output
            if preferred_output and preferred_output in output_names
            else output_names[0]
        )

        return {
            **state,
            "enabled": True,
            "ncnn": ncnn,
            "net": net,
            "input_blob": input_names[0],
            "output_blob": output_blob,
            "output_heads": output_names,
            "param_path": str(param_path),
            "bin_path": str(bin_path),
        }
    except Exception:
        return state


def normalize_embedding_vector(values: list[float]) -> list[float] | None:
    finite_values: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        finite_values.append(numeric)

    if not finite_values:
        return None

    norm = math.sqrt(sum(v * v for v in finite_values))
    if norm <= 1e-9:
        return None
    return [float(v / norm) for v in finite_values]


def embedding_cosine_distance(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or not a:
        return None

    dot = 0.0
    for left, right in zip(a, b):
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None
        lval = float(left)
        rval = float(right)
        if not math.isfinite(lval) or not math.isfinite(rval):
            return None
        dot += lval * rval

    similarity = max(-1.0, min(1.0, dot))
    return float(max(0.0, 1.0 - similarity))


def match_embedding_identity(
    identity_state: dict[str, Any],
    embedding: list[float],
    distance_threshold: float,
    timestamp_ms: int | None = None,
    split_guard_ratio: float = DEFAULT_IDENTITY_SPLIT_GUARD_RATIO,
    split_guard_max_seen: int = DEFAULT_IDENTITY_SPLIT_GUARD_MAX_SEEN,
    merge_recover_threshold: float | None = DEFAULT_IDENTITY_MERGE_RECOVER_THRESHOLD,
    merge_recover_min_seen: int = DEFAULT_IDENTITY_MERGE_RECOVER_MIN_SEEN,
    reserved_identity_ids: set[str] | None = None,
    verified_allowlist_mode: bool = False,
    audit: dict[str, Any] | None = None,
) -> tuple[str, float | None, bool]:
    normalized_embedding = normalize_embedding_vector(embedding)
    embedding_is_normalized = normalized_embedding is not None
    if normalized_embedding is None:
        normalized_embedding = [
            float(value)
            for value in embedding
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
    if not normalized_embedding:
        normalized_embedding = [0.0]

    profiles_raw = identity_state.get("profiles")
    profiles: dict[str, dict[str, Any]]
    if isinstance(profiles_raw, dict):
        profiles = profiles_raw
    else:
        profiles = {}
        identity_state["profiles"] = profiles

    candidates: list[tuple[str, float, int, dict[str, Any]]] = []
    for identity_id, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        stored = profile.get("embedding")
        if not isinstance(stored, list):
            continue
        distance = embedding_cosine_distance(normalized_embedding, stored)
        if distance is None:
            continue
        seen_raw = profile.get("seen", 0)
        seen = int(seen_raw) if isinstance(seen_raw, (int, float)) else 0
        candidates.append((str(identity_id), float(distance), max(0, seen), profile))

    candidates.sort(key=lambda item: item[1])

    threshold = max(0.0, min(2.0, float(distance_threshold)))
    split_ratio = max(0.0, min(1.0, float(split_guard_ratio)))
    split_seen_limit = max(0, int(split_guard_max_seen))
    merge_threshold = (
        max(threshold, min(2.0, float(merge_recover_threshold)))
        if isinstance(merge_recover_threshold, (int, float))
        and math.isfinite(float(merge_recover_threshold))
        else threshold
    )
    merge_seen_min = max(1, int(merge_recover_min_seen))
    reserved = set(reserved_identity_ids or set())

    policy = _normalize_uiface_policy(identity_state.get("uifacePolicy"))
    policy_instance_to_subject = _as_string_map(policy.get("instanceToSubjectIdMap"))
    policy_relation_map = _as_relation_map(policy.get("enrolledSubjectRelationMap"))
    policy_ignored_subjects = _expand_ignored_subjects(
        _as_string_set(policy.get("ignoredSubjectIds")),
        policy_relation_map,
    )
    policy_ignored_instances = _as_string_set(policy.get("ignoredInstanceIds"))
    policy_verified_unique_ids = _as_string_set(policy.get("verifiedUniqueIds"))
    enforce_verified_allowlist = bool(verified_allowlist_mode) and bool(
        policy_verified_unique_ids
    )

    split_guard_triggered = False
    reserved_blocked = False

    def _write_audit(
        decision: str,
        *,
        matched_identity_id: str | None,
        distance: float | None,
    ) -> None:
        if not isinstance(audit, dict):
            return
        audit["decision"] = str(decision)
        audit["matchedIdentityId"] = (
            str(matched_identity_id) if isinstance(matched_identity_id, str) else None
        )
        audit["distance"] = (
            float(distance)
            if isinstance(distance, (int, float)) and math.isfinite(float(distance))
            else None
        )
        audit["distanceThreshold"] = float(threshold)
        audit["mergeRecoverThreshold"] = float(merge_threshold)

    chosen: tuple[str, float, int, dict[str, Any]] | None = None
    chosen_reason = "strict_match"
    for identity_id, candidate_distance, candidate_seen, profile in candidates:
        if candidate_distance > threshold:
            break
        if enforce_verified_allowlist and identity_id not in policy_verified_unique_ids:
            reserved_blocked = True
            continue
        candidate_subject_id = policy_instance_to_subject.get(identity_id)
        if identity_id in policy_ignored_instances:
            reserved_blocked = True
            continue
        if (
            isinstance(candidate_subject_id, str)
            and candidate_subject_id
            and candidate_subject_id in policy_ignored_subjects
        ):
            reserved_blocked = True
            continue
        if identity_id in reserved:
            reserved_blocked = True
            continue
        split_guard_floor = threshold * split_ratio
        if (
            split_seen_limit > 0
            and candidate_seen <= split_seen_limit
            and candidate_distance > split_guard_floor
        ):
            split_guard_triggered = True
            continue
        chosen = (identity_id, candidate_distance, candidate_seen, profile)
        chosen_reason = "strict_match"
        break

    if chosen is None and merge_threshold > threshold:
        for identity_id, candidate_distance, candidate_seen, profile in candidates:
            if candidate_distance > merge_threshold:
                break
            if (
                enforce_verified_allowlist
                and identity_id not in policy_verified_unique_ids
            ):
                reserved_blocked = True
                continue
            candidate_subject_id = policy_instance_to_subject.get(identity_id)
            if identity_id in policy_ignored_instances:
                reserved_blocked = True
                continue
            if (
                isinstance(candidate_subject_id, str)
                and candidate_subject_id
                and candidate_subject_id in policy_ignored_subjects
            ):
                reserved_blocked = True
                continue
            if identity_id in reserved:
                reserved_blocked = True
                continue
            if candidate_seen < merge_seen_min:
                continue
            chosen = (identity_id, candidate_distance, candidate_seen, profile)
            chosen_reason = "merge_recover"
            break

    if chosen is not None:
        best_id, best_distance, _best_seen, profile = chosen
        seen = int(profile.get("seen", 0)) + 1
        stored = profile.get("embedding")
        updated = normalized_embedding
        if isinstance(stored, list) and len(stored) == len(normalized_embedding):
            momentum = (seen - 1) / float(max(1, seen))
            mixed = [
                (float(stored[idx]) * momentum)
                + (float(normalized_embedding[idx]) * (1.0 - momentum))
                for idx in range(len(normalized_embedding))
            ]
            normalized = normalize_embedding_vector(mixed)
            if normalized is not None:
                updated = normalized

        profiles[best_id] = {
            "embedding": updated,
            "seen": seen,
            "lastSeenMs": (
                int(timestamp_ms)
                if isinstance(timestamp_ms, int)
                else int(profile.get("lastSeenMs", 0))
            ),
        }
        identity_state["profiles"] = profiles
        _write_audit(
            chosen_reason,
            matched_identity_id=best_id,
            distance=float(best_distance),
        )
        return best_id, float(best_distance), False

    next_id = int(identity_state.get("next_id", 1))
    if next_id < 1:
        next_id = 1
    while True:
        candidate_identity = f"person-{next_id}"
        if (
            candidate_identity not in profiles
            and candidate_identity not in policy_ignored_instances
        ):
            new_identity = candidate_identity
            break
        next_id += 1

    profiles[new_identity] = {
        "embedding": list(normalized_embedding),
        "seen": 1,
        "lastSeenMs": int(timestamp_ms) if isinstance(timestamp_ms, int) else 0,
    }
    identity_state["profiles"] = profiles
    identity_state["next_id"] = next_id + 1

    if reserved_blocked:
        new_reason = "new_identity_reserved"
    elif split_guard_triggered:
        new_reason = "new_identity_split_guard"
    elif not candidates:
        new_reason = (
            "new_identity_no_profiles"
            if embedding_is_normalized
            else "new_identity_invalid_embedding"
        )
    elif not embedding_is_normalized:
        new_reason = "new_identity_invalid_embedding"
    else:
        new_reason = "new_identity_threshold"

    _write_audit(
        new_reason,
        matched_identity_id=None,
        distance=None,
    )
    return new_identity, None, True


def _empty_identity_state() -> dict[str, Any]:
    return {"next_id": 1, "profiles": {}}


def _identity_numeric_suffix(identity_id: str) -> int | None:
    value = str(identity_id)
    if not value.startswith("person-"):
        return None
    suffix = value[7:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _as_string_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if isinstance(value, str) and value}


def _as_string_map(values: Any) -> dict[str, str]:
    if not isinstance(values, dict):
        return {}
    mapped: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if not key or not value:
            continue
        mapped[str(key)] = str(value)
    return mapped


def _as_relation_map(values: Any) -> dict[str, list[str]]:
    if not isinstance(values, dict):
        return {}
    relation_map: dict[str, list[str]] = {}
    for key, raw_links in values.items():
        if not isinstance(key, str) or not key:
            continue
        links: list[str] = []
        if isinstance(raw_links, list):
            for item in raw_links:
                if isinstance(item, str) and item:
                    links.append(str(item))
        relation_map[str(key)] = sorted(set(links))
    return relation_map


def _normalize_uiface_policy(policy_raw: Any) -> dict[str, Any]:
    if not isinstance(policy_raw, dict):
        return {
            "instanceToSubjectIdMap": {},
            "enrolledSubjectRelationMap": {},
            "ignoredSubjectIds": [],
            "ignoredInstanceIds": [],
            "verifiedUniqueIds": [],
        }

    return {
        "instanceToSubjectIdMap": _as_string_map(
            policy_raw.get("instanceToSubjectIdMap")
        ),
        "enrolledSubjectRelationMap": _as_relation_map(
            policy_raw.get("enrolledSubjectRelationMap")
        ),
        "ignoredSubjectIds": sorted(
            _as_string_set(policy_raw.get("ignoredSubjectIds"))
        ),
        "ignoredInstanceIds": sorted(
            _as_string_set(policy_raw.get("ignoredInstanceIds"))
        ),
        "verifiedUniqueIds": sorted(
            _as_string_set(policy_raw.get("verifiedUniqueIds"))
        ),
    }


def _extract_uiface_policy_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _normalize_uiface_policy(
        {
            "instanceToSubjectIdMap": payload.get("InstanceIdToSubjectIdMap"),
            "enrolledSubjectRelationMap": payload.get("EnrolledSubjectRelationMap"),
            "ignoredSubjectIds": payload.get("IgnoredSubjectIds"),
            "ignoredInstanceIds": payload.get("IngnoredInstanceIds"),
            "verifiedUniqueIds": payload.get("VerifiedUniqueIds"),
        }
    )


def _uiface_policy_has_entries(policy: dict[str, Any]) -> bool:
    return bool(
        policy.get("instanceToSubjectIdMap")
        or policy.get("enrolledSubjectRelationMap")
        or policy.get("ignoredSubjectIds")
        or policy.get("ignoredInstanceIds")
        or policy.get("verifiedUniqueIds")
    )


def _expand_ignored_subjects(
    ignored_subject_ids: set[str],
    relation_map: dict[str, list[str]],
) -> set[str]:
    if not ignored_subject_ids or not relation_map:
        return set(ignored_subject_ids)

    expanded = set(ignored_subject_ids)
    changed = True
    while changed:
        changed = False
        for subject_id, related_ids in relation_map.items():
            related = {
                str(value) for value in related_ids if isinstance(value, str) and value
            }
            if not related:
                continue
            if subject_id in expanded or bool(expanded.intersection(related)):
                before = len(expanded)
                expanded.add(subject_id)
                expanded.update(related)
                if len(expanded) != before:
                    changed = True
    return expanded


def _sanitize_identity_state(
    state: dict[str, Any],
    max_profiles: int,
) -> dict[str, Any]:
    limit = max(1, int(max_profiles))
    profiles_raw = state.get("profiles")
    profiles: dict[str, dict[str, Any]] = {}
    entries: list[tuple[str, list[float], int, int]] = []

    if isinstance(profiles_raw, dict):
        profile_items = profiles_raw.items()
    elif isinstance(profiles_raw, list):
        profile_items = []
        for entry in profiles_raw:
            if not isinstance(entry, dict):
                continue
            identity_id = entry.get("identityId")
            if not isinstance(identity_id, str):
                continue
            profile_items.append((identity_id, entry))
    else:
        profile_items = []

    for identity_id, profile in profile_items:
        if not isinstance(identity_id, str) or not isinstance(profile, dict):
            continue

        embedding = profile.get("embedding")
        if not isinstance(embedding, list):
            continue
        normalized = normalize_embedding_vector(embedding)
        if normalized is None:
            continue

        seen_raw = profile.get("seen", 1)
        seen = int(seen_raw) if isinstance(seen_raw, (int, float)) else 1
        seen = max(1, seen)

        last_seen_raw = profile.get("lastSeenMs", 0)
        last_seen = (
            int(last_seen_raw)
            if isinstance(last_seen_raw, (int, float))
            and math.isfinite(float(last_seen_raw))
            else 0
        )

        entries.append((str(identity_id), normalized, seen, last_seen))

    entries.sort(key=lambda item: (item[2], item[3]), reverse=True)
    for identity_id, embedding, seen, last_seen in entries[:limit]:
        profiles[identity_id] = {
            "embedding": embedding,
            "seen": seen,
            "lastSeenMs": last_seen,
        }

    next_id_raw = state.get("next_id")
    if not isinstance(next_id_raw, int) or next_id_raw < 1:
        next_id_raw = 1
    max_suffix = 0
    for identity_id in profiles.keys():
        suffix = _identity_numeric_suffix(identity_id)
        if suffix is not None and suffix > max_suffix:
            max_suffix = suffix
    next_id = max(int(next_id_raw), max_suffix + 1)

    return {"next_id": next_id, "profiles": profiles}


def _build_uiface_identity_indexes(state: dict[str, Any]) -> dict[str, Any]:
    profiles = state.get("profiles")
    if not isinstance(profiles, dict):
        return {
            "EnrolledSubjectIds": [],
            "EnrolledSubjectRelationMap": {},
            "IgnoredSubjectIds": [],
            "IngnoredInstanceIds": [],
            "InstanceIdToNameMap": {},
            "InstanceIdToSubjectIdMap": {},
            "InstanceIdToUniqueIdMap": {},
            "SubjectIdToNameMap": {},
            "VerifiedUniqueIds": [],
        }

    policy = _normalize_uiface_policy(state.get("uifacePolicy"))

    instance_to_name: dict[str, str] = {}
    instance_to_subject: dict[str, str] = {}
    instance_to_unique: dict[str, str] = {}
    subject_to_name: dict[str, str] = {}
    enrolled_subject_ids: list[str] = []
    verified_unique_ids: list[str] = []

    for identity_id in sorted(str(key) for key in profiles.keys()):
        subject_id = f"subject-{identity_id}"
        unique_id = identity_id
        display_name = identity_id
        instance_to_name[identity_id] = display_name
        instance_to_subject[identity_id] = subject_id
        instance_to_unique[identity_id] = unique_id
        subject_to_name[subject_id] = display_name
        enrolled_subject_ids.append(subject_id)
        verified_unique_ids.append(unique_id)

    policy_relation_map = _as_relation_map(policy.get("enrolledSubjectRelationMap"))
    ignored_subject_ids = sorted(_as_string_set(policy.get("ignoredSubjectIds")))
    ignored_instance_ids = sorted(_as_string_set(policy.get("ignoredInstanceIds")))
    verified_from_policy = _as_string_set(policy.get("verifiedUniqueIds"))
    if verified_from_policy:
        verified_unique_ids = sorted(verified_from_policy)

    return {
        "EnrolledSubjectIds": enrolled_subject_ids,
        "EnrolledSubjectRelationMap": policy_relation_map,
        "IgnoredSubjectIds": ignored_subject_ids,
        "IngnoredInstanceIds": ignored_instance_ids,
        "InstanceIdToNameMap": instance_to_name,
        "InstanceIdToSubjectIdMap": instance_to_subject,
        "InstanceIdToUniqueIdMap": instance_to_unique,
        "SubjectIdToNameMap": subject_to_name,
        "VerifiedUniqueIds": verified_unique_ids,
    }


def _infer_next_id_from_uiface_payload(payload: dict[str, Any]) -> int:
    instance_to_unique = payload.get("InstanceIdToUniqueIdMap")
    if not isinstance(instance_to_unique, dict):
        return 1

    max_suffix = 0
    for value in instance_to_unique.values():
        if not isinstance(value, str):
            continue
        suffix = _identity_numeric_suffix(value)
        if suffix is not None and suffix > max_suffix:
            max_suffix = suffix

    return max(1, max_suffix + 1)


def load_identity_gallery(
    gallery_path: str | None,
    max_profiles: int = DEFAULT_IDENTITY_GALLERY_MAX_PROFILES,
) -> dict[str, Any]:
    if not gallery_path or not str(gallery_path).strip():
        return _empty_identity_state()

    path = Path(str(gallery_path)).resolve()
    if not path.is_file():
        return _empty_identity_state()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_identity_state()

    if not isinstance(payload, dict):
        return _empty_identity_state()

    if "profiles" in payload and "next_id" in payload:
        state = _sanitize_identity_state(payload, max_profiles=max_profiles)
        policy = _extract_uiface_policy_from_payload(payload)
        if _uiface_policy_has_entries(policy):
            state["uifacePolicy"] = policy
        return state

    if "profiles" not in payload and isinstance(
        payload.get("InstanceIdToUniqueIdMap"), dict
    ):
        state = _sanitize_identity_state(
            {
                "next_id": _infer_next_id_from_uiface_payload(payload),
                "profiles": [],
            },
            max_profiles=max_profiles,
        )
        policy = _extract_uiface_policy_from_payload(payload)
        if _uiface_policy_has_entries(policy):
            state["uifacePolicy"] = policy
        return state

    next_id = payload.get("nextId", 1)
    profiles_list = payload.get("profiles", [])
    transformed = {
        "next_id": int(next_id) if isinstance(next_id, int) else 1,
        "profiles": profiles_list,
    }
    state = _sanitize_identity_state(transformed, max_profiles=max_profiles)
    policy = _extract_uiface_policy_from_payload(payload)
    if _uiface_policy_has_entries(policy):
        state["uifacePolicy"] = policy
    return state


def save_identity_gallery(
    gallery_path: str | None,
    identity_state: dict[str, Any],
    max_profiles: int = DEFAULT_IDENTITY_GALLERY_MAX_PROFILES,
) -> bool:
    if not gallery_path or not str(gallery_path).strip():
        return False

    path = Path(str(gallery_path)).resolve()
    state = _sanitize_identity_state(identity_state, max_profiles=max_profiles)
    policy = _normalize_uiface_policy(identity_state.get("uifacePolicy"))
    if _uiface_policy_has_entries(policy):
        state["uifacePolicy"] = policy

    profiles_payload: list[dict[str, Any]] = []
    for identity_id, profile in state.get("profiles", {}).items():
        if not isinstance(identity_id, str) or not isinstance(profile, dict):
            continue
        embedding = profile.get("embedding")
        if not isinstance(embedding, list):
            continue
        seen = profile.get("seen", 1)
        last_seen = profile.get("lastSeenMs", 0)
        profiles_payload.append(
            {
                "identityId": identity_id,
                "embedding": [
                    float(v) for v in embedding if isinstance(v, (int, float))
                ],
                "seen": int(seen) if isinstance(seen, (int, float)) else 1,
                "lastSeenMs": (
                    int(last_seen)
                    if isinstance(last_seen, (int, float))
                    and math.isfinite(float(last_seen))
                    else 0
                ),
            }
        )

    payload = {
        "version": 1,
        "savedAtMs": int(time.time() * 1000),
        "nextId": int(state.get("next_id", 1)),
        "profiles": profiles_payload,
    }
    payload.update(_build_uiface_identity_indexes(state))

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp-{time.time_ns()}")
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)
        return True
    except Exception:
        return False


def prune_identity_gallery(
    identity_state: dict[str, Any],
    now_ms: int,
    max_idle_ms: int,
) -> list[str]:
    limit = int(max_idle_ms)
    if limit <= 0:
        return []

    profiles_raw = identity_state.get("profiles")
    if not isinstance(profiles_raw, dict):
        return []

    now = int(now_ms)
    removed_ids: list[str] = []
    for identity_id in list(profiles_raw.keys()):
        profile = profiles_raw.get(identity_id)
        if not isinstance(profile, dict):
            continue

        last_seen_raw = profile.get("lastSeenMs")
        if not isinstance(last_seen_raw, (int, float)) or not math.isfinite(
            float(last_seen_raw)
        ):
            continue
        last_seen = int(last_seen_raw)
        if last_seen <= 0:
            continue
        if (now - last_seen) <= limit:
            continue

        removed_ids.append(str(identity_id))
        profiles_raw.pop(identity_id, None)

    identity_state["profiles"] = profiles_raw
    return removed_ids


def evaluate_pose_mask_quality(
    pose_angles: tuple[float, float, float] | None,
    mask_confidence: float | None,
    verifier_enabled: bool,
    hard_gate_enabled: bool,
    max_abs_yaw: float,
    max_abs_pitch: float,
    max_abs_roll: float,
    mask_confidence_min: float,
) -> tuple[bool, str, dict[str, Any]]:
    pose_available = False
    yaw = 0.0
    pitch = 0.0
    roll = 0.0
    if (
        isinstance(pose_angles, tuple)
        and len(pose_angles) == 3
        and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in pose_angles
        )
    ):
        pose_available = True
        yaw = float(pose_angles[0])
        pitch = float(pose_angles[1])
        roll = float(pose_angles[2])

    mask_available = isinstance(mask_confidence, (int, float)) and math.isfinite(
        float(mask_confidence)
    )
    mask_value = 0.0
    if mask_available and isinstance(mask_confidence, (int, float)):
        mask_value = float(mask_confidence)
    pose_out_of_range = pose_available and (
        abs(yaw) > float(max_abs_yaw)
        or abs(pitch) > float(max_abs_pitch)
        or abs(roll) > float(max_abs_roll)
    )
    mask_low_confidence = mask_available and (mask_value < float(mask_confidence_min))

    details = {
        "poseUnavailable": not pose_available,
        "maskUnavailable": not mask_available,
        "poseOutOfRange": pose_out_of_range,
        "maskLowConfidence": mask_low_confidence,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "maskConfidence": mask_value,
    }

    if not verifier_enabled:
        return True, "quality_verifier_disabled", details

    if not pose_available:
        return (not hard_gate_enabled), "pose_unavailable", details

    if not mask_available:
        return (not hard_gate_enabled), "mask_unavailable", details

    if hard_gate_enabled:
        if pose_out_of_range:
            return False, "pose_out_of_range", details
        if mask_low_confidence:
            return False, "mask_confidence_low", details
        return True, "quality_ok", details

    if pose_out_of_range or mask_low_confidence:
        return True, "quality_ok_telemetry_only", details
    return True, "quality_ok", details


def evaluate_identity_candidate_quality(
    face_detection: dict[str, Any],
    frame_area: int,
    landmarks: list[tuple[float, float]] | None,
    min_face_score: float,
    min_face_area_ratio: float,
    require_landmarks: bool,
) -> tuple[bool, str | None]:
    score = face_detection.get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        return False, "low_score"
    if float(score) < float(min_face_score):
        return False, "low_score"

    bbox = face_detection.get("bbox")
    if not isinstance(bbox, tuple) or len(bbox) != 4:
        return False, "invalid_bbox"
    x, y, w, h = bbox
    if any(not isinstance(value, (int, float)) for value in (x, y, w, h)):
        return False, "invalid_bbox"

    area_ratio = (float(w) * float(h)) / float(max(1, int(frame_area)))
    if area_ratio < float(min_face_area_ratio):
        return False, "small_face"

    if require_landmarks:
        if not isinstance(landmarks, list) or len(landmarks) < 3:
            return False, "missing_landmarks"

    return True, None


def landmark_guided_bbox(
    landmarks: list[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> Box | None:
    valid_points: list[tuple[float, float]] = []
    for point in landmarks:
        if not isinstance(point, tuple) or len(point) != 2:
            continue
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        nx = float(x)
        ny = float(y)
        if not math.isfinite(nx) or not math.isfinite(ny):
            continue
        if nx < 0.0 or nx > 1.0 or ny < 0.0 or ny > 1.0:
            continue
        valid_points.append((nx, ny))

    if len(valid_points) < 3:
        return None

    min_x = min(point[0] for point in valid_points)
    max_x = max(point[0] for point in valid_points)
    min_y = min(point[1] for point in valid_points)
    max_y = max(point[1] for point in valid_points)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    span_x = max(0.15, (max_x - min_x) * 2.2)
    span_y = max(0.18, (max_y - min_y) * 2.6)

    left = max(0.0, center_x - (span_x / 2.0))
    top = max(0.0, center_y - (span_y / 2.0))
    right = min(1.0, center_x + (span_x / 2.0))
    bottom = min(1.0, center_y + (span_y / 2.0))

    width_px = max(1, int(round((right - left) * max(1, int(image_width)))))
    height_px = max(1, int(round((bottom - top) * max(1, int(image_height)))))
    x_px = max(0, min(max(1, int(image_width)) - 1, int(round(left * image_width))))
    y_px = max(0, min(max(1, int(image_height)) - 1, int(round(top * image_height))))
    if x_px + width_px > image_width:
        width_px = image_width - x_px
    if y_px + height_px > image_height:
        height_px = image_height - y_px
    if width_px <= 0 or height_px <= 0:
        return None

    return (x_px, y_px, width_px, height_px)


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
    smartface_lmk_param: str | None = None,
    smartface_lmk_bin: str | None = None,
    smartface_extract_param: str | None = None,
    smartface_extract_bin: str | None = None,
    smartface_identity_distance_threshold: float = 0.35,
    smartface_identity_stable_frames: int = 2,
    smartface_identity_min_face_score: float = 0.75,
    smartface_identity_min_face_area_ratio: float = 0.0012,
    smartface_identity_require_landmarks: bool = False,
    smartface_identity_split_guard_ratio: float = DEFAULT_IDENTITY_SPLIT_GUARD_RATIO,
    smartface_identity_split_guard_max_seen: int = DEFAULT_IDENTITY_SPLIT_GUARD_MAX_SEEN,
    smartface_identity_merge_recover_threshold: float = DEFAULT_IDENTITY_MERGE_RECOVER_THRESHOLD,
    smartface_identity_merge_recover_min_seen: int = DEFAULT_IDENTITY_MERGE_RECOVER_MIN_SEEN,
    smartface_identity_prevent_duplicate_per_frame: bool = True,
    smartface_identity_verified_allowlist_mode: bool = False,
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
            "output_heads": tuple(),
            "enroll_enabled": False,
        }

        try:
            ncnn = importlib.import_module("ncnn")
            net = ncnn.Net()
            rc_param = int(net.load_param(str(param_path)))
            rc_model = int(net.load_model(str(bin_path)))
            if rc_param == 0 and rc_model == 0:
                input_names = list(net.input_names())
                output_names = tuple(str(name) for name in list(net.output_names()))
                enroll_enabled = set(ENROLL_REQUIRED_OUTPUTS).issubset(
                    set(output_names)
                )
                ncnn_state = {
                    "enabled": bool(input_names and output_names),
                    "net": net,
                    "ncnn": ncnn,
                    "input_blob": input_names[0] if input_names else None,
                    "output_blob": output_names[0] if output_names else None,
                    "output_heads": output_names,
                    "enroll_enabled": bool(enroll_enabled),
                }

                lmk_pair = _resolve_model_pair(
                    explicit_param=(
                        str(smartface_lmk_param) if smartface_lmk_param else None
                    ),
                    explicit_bin=(
                        str(smartface_lmk_bin) if smartface_lmk_bin else None
                    ),
                    base_dir=param_path.parent,
                    param_candidates=SMARTFACE_LMK_PARAM_CANDIDATES,
                    bin_candidates=SMARTFACE_LMK_BIN_CANDIDATES,
                )
                extract_pair = _resolve_model_pair(
                    explicit_param=(
                        str(smartface_extract_param)
                        if smartface_extract_param
                        else None
                    ),
                    explicit_bin=(
                        str(smartface_extract_bin) if smartface_extract_bin else None
                    ),
                    base_dir=param_path.parent,
                    param_candidates=SMARTFACE_EXTRACT_PARAM_CANDIDATES,
                    bin_candidates=SMARTFACE_EXTRACT_BIN_CANDIDATES,
                )
                lmk_state = _load_optional_ncnn_model(
                    ncnn=ncnn,
                    model_pair=lmk_pair,
                    input_size=112,
                    pixel_type="PIXEL_BGR2RGB",
                    mean_vals=[127.5, 127.5, 127.5],
                    norm_vals=[1.0 / 128.0, 1.0 / 128.0, 1.0 / 128.0],
                )
                extract_state = _load_optional_ncnn_model(
                    ncnn=ncnn,
                    model_pair=extract_pair,
                    input_size=max(16, int(smartface_input_size)),
                    pixel_type="PIXEL_BGR2RGB",
                    mean_vals=[127.5, 127.5, 127.5],
                    norm_vals=[1.0 / 128.0, 1.0 / 128.0, 1.0 / 128.0],
                    preferred_output="embedding",
                )
            else:
                lmk_state = _load_optional_ncnn_model(
                    ncnn=None,
                    model_pair=None,
                    input_size=112,
                    pixel_type="PIXEL_BGR2RGB",
                    mean_vals=[127.5, 127.5, 127.5],
                    norm_vals=[1.0 / 128.0, 1.0 / 128.0, 1.0 / 128.0],
                )
                extract_state = _load_optional_ncnn_model(
                    ncnn=None,
                    model_pair=None,
                    input_size=max(16, int(smartface_input_size)),
                    pixel_type="PIXEL_BGR2RGB",
                    mean_vals=[127.5, 127.5, 127.5],
                    norm_vals=[1.0 / 128.0, 1.0 / 128.0, 1.0 / 128.0],
                    preferred_output="embedding",
                )
        except Exception:
            ncnn_state = {
                "enabled": False,
                "net": None,
                "ncnn": None,
                "input_blob": None,
                "output_blob": None,
                "output_heads": tuple(),
                "enroll_enabled": False,
            }
            lmk_state = {
                "enabled": False,
                "net": None,
                "input_blob": None,
                "output_blob": None,
                "output_heads": tuple(),
                "input_size": 112,
                "pixel_type": "PIXEL_BGR2RGB",
                "mean_vals": [127.5, 127.5, 127.5],
                "norm_vals": [1.0 / 128.0, 1.0 / 128.0, 1.0 / 128.0],
            }
            extract_state = {
                "enabled": False,
                "net": None,
                "input_blob": None,
                "output_blob": None,
                "output_heads": tuple(),
                "input_size": max(16, int(smartface_input_size)),
                "pixel_type": "PIXEL_BGR2RGB",
                "mean_vals": [127.5, 127.5, 127.5],
                "norm_vals": [1.0 / 128.0, 1.0 / 128.0, 1.0 / 128.0],
            }

        return {
            "name": "smartface_ncnn",
            "face_cascade": cascade,
            "min_score": max(0.0, float(smartface_min_score)),
            "input_size": max(16, int(smartface_input_size)),
            "nms_threshold": float(max(0.0, min(1.0, nms_threshold))),
            "identity_distance_threshold": float(
                max(0.0, min(2.0, smartface_identity_distance_threshold))
            ),
            "identity_stable_frames": max(1, int(smartface_identity_stable_frames)),
            "identity_min_face_score": float(
                max(0.0, min(1.0, smartface_identity_min_face_score))
            ),
            "identity_min_face_area_ratio": float(
                max(0.0, min(1.0, smartface_identity_min_face_area_ratio))
            ),
            "identity_require_landmarks": bool(smartface_identity_require_landmarks),
            "identity_split_guard_ratio": float(
                max(0.0, min(1.0, smartface_identity_split_guard_ratio))
            ),
            "identity_split_guard_max_seen": max(
                0, int(smartface_identity_split_guard_max_seen)
            ),
            "identity_merge_recover_threshold": float(
                max(0.0, min(2.0, smartface_identity_merge_recover_threshold))
            ),
            "identity_merge_recover_min_seen": max(
                1, int(smartface_identity_merge_recover_min_seen)
            ),
            "identity_prevent_duplicate_per_frame": bool(
                smartface_identity_prevent_duplicate_per_frame
            ),
            "identity_verified_allowlist_mode": bool(
                smartface_identity_verified_allowlist_mode
            ),
            "lmk_model": lmk_state,
            "extract_model": extract_state,
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
    if str(output_blob) != "embedding":
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


def _frame_dimensions(frame_bgr: Any, fallback: int) -> tuple[int, int]:
    shape = getattr(frame_bgr, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 2:
        height = shape[0]
        width = shape[1]
        if isinstance(height, (int, float)) and isinstance(width, (int, float)):
            resolved_h = max(1, int(height))
            resolved_w = max(1, int(width))
            return resolved_h, resolved_w

    default_dim = max(1, int(fallback))
    return default_dim, default_dim


def _build_enroll_anchor_sizes(
    stride: int,
    input_size: int,
    anchors_per_cell: int,
) -> list[tuple[float, float]]:
    scale_hints = list(ENROLL_ANCHOR_SCALE_HINTS.get(int(stride), (1.0,)))
    if len(scale_hints) < anchors_per_cell:
        tail = scale_hints[-1] if scale_hints else 1.0
        while len(scale_hints) < anchors_per_cell:
            scale_hints.append(tail)

    base = float(stride) / float(max(1, int(input_size)))
    sizes: list[tuple[float, float]] = []
    for idx in range(anchors_per_cell):
        scale = float(scale_hints[idx])
        width = max(0.05, min(0.98, base * scale))
        height = max(0.05, min(0.98, width * 1.18))
        sizes.append((width, height))

    return sizes


def _face_probability(bg_value: float, face_value: float) -> float:
    bg = float(bg_value)
    face = float(face_value)
    if 0.0 <= bg <= 1.0 and 0.0 <= face <= 1.0 and abs((bg + face) - 1.0) <= 0.10:
        return max(0.0, min(1.0, face))

    peak = max(bg, face)
    e_bg = math.exp(bg - peak)
    e_face = math.exp(face - peak)
    denom = e_bg + e_face
    if denom <= 0.0:
        return 0.0
    return max(0.0, min(1.0, e_face / denom))


def run_smartface_enroll_forward(
    detector: dict[str, Any],
    frame_bgr: Any,
) -> dict[str, dict[str, Any]]:
    ncnn_state = detector.get("ncnn") or {}
    if not ncnn_state.get("enabled"):
        return {}

    output_heads = tuple(str(name) for name in (ncnn_state.get("output_heads") or ()))
    if not set(ENROLL_REQUIRED_OUTPUTS).issubset(set(output_heads)):
        return {}

    ncnn = ncnn_state.get("ncnn")
    net = ncnn_state.get("net")
    input_blob = ncnn_state.get("input_blob")
    if not ncnn or not net or not input_blob:
        return {}

    input_size = max(16, int(detector.get("input_size", 112)))

    try:
        resized = cv2.resize(
            frame_bgr,
            (input_size, input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mat = ncnn.Mat.from_pixels(
            resized,
            ncnn.Mat.PixelType.PIXEL_BGR,
            int(input_size),
            int(input_size),
        )

        ex = net.create_extractor()
        if int(ex.input(input_blob, mat)) != 0:
            return {}

        outputs: dict[str, dict[str, Any]] = {}
        for head_name in ENROLL_REQUIRED_OUTPUTS:
            extract_rc, out_mat = ex.extract(head_name)
            if int(extract_rc) != 0 or out_mat is None:
                return {}

            values = out_mat.numpy("f").reshape(-1)
            outputs[head_name] = {
                "w": int(getattr(out_mat, "w", 0)),
                "h": int(getattr(out_mat, "h", 0)),
                "c": int(getattr(out_mat, "c", 0)),
                "data": [float(v) for v in values],
            }

        return outputs
    except Exception:
        return {}


def decode_enroll_face_detections(
    head_outputs: dict[str, dict[str, Any]],
    frame_width: int,
    frame_height: int,
    input_size: int,
    min_score: float,
    nms_threshold: float,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    score_floor = max(0.0, min(1.0, float(min_score)))
    resolved_nms = max(0.0, min(1.0, float(nms_threshold)))
    image_w = max(1, int(frame_width))
    image_h = max(1, int(frame_height))
    model_input = max(16, int(input_size))

    for stride in (16, 32, 64):
        cls_key = f"classification_stride_{stride}"
        reg_key = f"regression_stride_{stride}"
        cls_head = head_outputs.get(cls_key)
        reg_head = head_outputs.get(reg_key)
        if not isinstance(cls_head, dict) or not isinstance(reg_head, dict):
            continue

        cls_w = int(cls_head.get("w", 0))
        cls_h = int(cls_head.get("h", 0))
        cls_c = int(cls_head.get("c", 0))
        reg_w = int(reg_head.get("w", 0))
        reg_h = int(reg_head.get("h", 0))
        reg_c = int(reg_head.get("c", 0))
        if (
            cls_w <= 0
            or cls_h <= 0
            or cls_c < 2
            or reg_w <= 0
            or reg_h <= 0
            or reg_c < 4
        ):
            continue

        cls_values_raw = cls_head.get("data")
        reg_values_raw = reg_head.get("data")
        if not isinstance(cls_values_raw, list) or not isinstance(reg_values_raw, list):
            continue

        cls_values: list[float] = []
        for value in cls_values_raw:
            if not isinstance(value, (int, float)):
                cls_values = []
                break
            numeric = float(value)
            if not math.isfinite(numeric):
                cls_values = []
                break
            cls_values.append(numeric)

        reg_values: list[float] = []
        for value in reg_values_raw:
            if not isinstance(value, (int, float)):
                reg_values = []
                break
            numeric = float(value)
            if not math.isfinite(numeric):
                reg_values = []
                break
            reg_values.append(numeric)

        if not cls_values or not reg_values:
            continue

        cls_anchor_count = cls_w * cls_h
        if len(cls_values) < cls_c * cls_anchor_count:
            continue

        spatial = reg_w * reg_h
        if spatial <= 0:
            continue
        anchors_per_cell = reg_c // 4
        if anchors_per_cell <= 0:
            continue
        if len(reg_values) < reg_c * spatial:
            continue

        expected_anchors = spatial * anchors_per_cell
        if expected_anchors <= 0:
            continue
        if expected_anchors > cls_anchor_count:
            expected_anchors = cls_anchor_count

        anchor_sizes = _build_enroll_anchor_sizes(
            stride=stride,
            input_size=model_input,
            anchors_per_cell=anchors_per_cell,
        )

        for anchor_idx in range(expected_anchors):
            bg_score = cls_values[anchor_idx]
            face_score = cls_values[cls_anchor_count + anchor_idx]
            confidence = _face_probability(bg_score, face_score)
            if confidence < score_floor:
                continue

            cell_idx = anchor_idx // anchors_per_cell
            anchor_slot = anchor_idx % anchors_per_cell
            cell_x = cell_idx % reg_w
            cell_y = cell_idx // reg_w
            if cell_y >= reg_h:
                continue

            prior_w, prior_h = anchor_sizes[min(anchor_slot, len(anchor_sizes) - 1)]
            prior_cx = ((float(cell_x) + 0.5) * float(stride)) / float(model_input)
            prior_cy = ((float(cell_y) + 0.5) * float(stride)) / float(model_input)

            reg_channel_offset = anchor_slot * 4
            dx = reg_values[(reg_channel_offset + 0) * spatial + cell_idx]
            dy = reg_values[(reg_channel_offset + 1) * spatial + cell_idx]
            dw = reg_values[(reg_channel_offset + 2) * spatial + cell_idx]
            dh = reg_values[(reg_channel_offset + 3) * spatial + cell_idx]

            pred_cx = prior_cx + (dx * ENROLL_VARIANCE_CENTER * prior_w)
            pred_cy = prior_cy + (dy * ENROLL_VARIANCE_CENTER * prior_h)
            pred_w = prior_w * math.exp(
                max(-10.0, min(10.0, dw * ENROLL_VARIANCE_SIZE))
            )
            pred_h = prior_h * math.exp(
                max(-10.0, min(10.0, dh * ENROLL_VARIANCE_SIZE))
            )

            pred_w = max(1e-4, min(1.0, pred_w))
            pred_h = max(1e-4, min(1.0, pred_h))
            left = max(0.0, pred_cx - (pred_w / 2.0))
            top = max(0.0, pred_cy - (pred_h / 2.0))
            width = max(1e-4, min(pred_w, 1.0 - left))
            height = max(1e-4, min(pred_h, 1.0 - top))

            px = max(0, min(image_w - 1, int(round(left * image_w))))
            py = max(0, min(image_h - 1, int(round(top * image_h))))
            pw = max(1, int(round(width * image_w)))
            ph = max(1, int(round(height * image_h)))
            if px + pw > image_w:
                pw = image_w - px
            if py + ph > image_h:
                ph = image_h - py
            if pw <= 0 or ph <= 0:
                continue

            detections.append(
                {
                    "class": "face",
                    "bbox": (px, py, pw, ph),
                    "score": float(max(0.0, min(0.99, confidence))),
                }
            )

    return dedupe_face_detections(detections, overlap_threshold=resolved_nms)


def _run_ncnn_vector(model_state: dict[str, Any], image_bgr: Any) -> list[float] | None:
    if not isinstance(model_state, dict) or not model_state.get("enabled"):
        return None

    ncnn = model_state.get("ncnn")
    if ncnn is None:
        try:
            ncnn = importlib.import_module("ncnn")
        except Exception:
            return None

    net = model_state.get("net")
    input_blob = model_state.get("input_blob")
    output_blob = model_state.get("output_blob")
    if not net or not input_blob or not output_blob:
        return None

    input_size = max(16, int(model_state.get("input_size", 112)))
    pixel_name = str(model_state.get("pixel_type") or "PIXEL_BGR2RGB")
    mean_vals = model_state.get("mean_vals")
    norm_vals = model_state.get("norm_vals")

    try:
        resized = cv2.resize(
            image_bgr,
            (input_size, input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        pixel_type = getattr(ncnn.Mat.PixelType, pixel_name)
        mat = ncnn.Mat.from_pixels(
            resized,
            pixel_type,
            int(input_size),
            int(input_size),
        )
        if isinstance(mean_vals, list) or isinstance(norm_vals, list):
            resolved_mean = (
                mean_vals if isinstance(mean_vals, list) else [0.0, 0.0, 0.0]
            )
            resolved_norm = (
                norm_vals if isinstance(norm_vals, list) else [1.0, 1.0, 1.0]
            )
            mat.substract_mean_normalize(resolved_mean, resolved_norm)

        ex = net.create_extractor()
        if int(ex.input(str(input_blob), mat)) != 0:
            return None
        extract_rc, out_mat = ex.extract(str(output_blob))
        if int(extract_rc) != 0 or out_mat is None:
            return None
        values = out_mat.numpy("f").reshape(-1)
        if values.size == 0:
            return None
        return [float(v) for v in values]
    except Exception:
        return None


def run_smartface_lmk_landmarks(
    detector: dict[str, Any],
    face_crop: Any,
) -> list[tuple[float, float]] | None:
    model_state = detector.get("lmk_model")
    if not isinstance(model_state, dict) or not model_state.get("enabled"):
        return None

    vector = _run_ncnn_vector(model_state, face_crop)
    if not isinstance(vector, list) or len(vector) < 10:
        return None

    input_size = float(max(16, int(model_state.get("input_size", 112))))
    absolute_like = max(abs(float(v)) for v in vector[:10]) > 2.0
    scale = (1.0 / input_size) if absolute_like else 1.0

    points: list[tuple[float, float]] = []
    for idx in range(5):
        x = float(vector[idx * 2]) * scale
        y = float(vector[idx * 2 + 1]) * scale
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        points.append((x, y))

    return points


def _slice_bbox(image_bgr: Any, bbox: Box) -> Any | None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    crop = image_bgr[y : y + h, x : x + w]
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    return crop


def extract_face_embedding(
    detector: dict[str, Any],
    frame_bgr: Any,
    face_bbox: Box,
) -> tuple[list[float] | None, list[tuple[float, float]] | None]:
    extract_state = detector.get("extract_model")
    if not isinstance(extract_state, dict) or not extract_state.get("enabled"):
        return None, None

    face_crop = _slice_bbox(frame_bgr, face_bbox)
    if face_crop is None:
        return None, None

    landmarks = run_smartface_lmk_landmarks(detector, face_crop)
    guided_bbox: Box | None = None
    if landmarks is not None:
        crop_h, crop_w = _frame_dimensions(face_crop, 112)
        guided_bbox = landmark_guided_bbox(
            landmarks=landmarks,
            image_width=crop_w,
            image_height=crop_h,
        )

    embedding_crop = _slice_bbox(face_crop, guided_bbox) if guided_bbox else face_crop
    if embedding_crop is None:
        embedding_crop = face_crop

    vector = _run_ncnn_vector(extract_state, embedding_crop)
    if not isinstance(vector, list):
        return None, landmarks
    normalized = normalize_embedding_vector(vector)
    return normalized, landmarks


def make_identity_event_payload(
    edge_type: str,
    event_id: str,
    identity_id: str,
    confidence: float,
    timestamp_ms: int,
    distance: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "functionName": "EventSmartDetectIdentity",
        "payload": {
            "edgeType": str(edge_type),
            "eventId": str(event_id),
            "eventType": "motion",
            "timestamp": int(timestamp_ms),
            "objectTypes": ["person"],
            "identityId": str(identity_id),
            "zonesStatus": [
                {
                    "zoneIds": [1],
                    "objectTypes": ["person"],
                    "confidence": round(float(confidence), 4),
                }
            ],
        },
    }

    if isinstance(distance, (int, float)) and math.isfinite(float(distance)):
        payload["payload"]["distance"] = round(float(distance), 6)

    return payload


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


def score_smartface_face_candidates(
    detector: dict[str, Any],
    frame_bgr: Any,
    candidate_boxes: list[Box],
) -> list[dict[str, Any]]:
    if detector.get("name") != "smartface_ncnn":
        return []

    input_size = max(16, int(detector.get("input_size", 112)))
    frame_h, frame_w = _frame_dimensions(frame_bgr, input_size)
    head_outputs = run_smartface_enroll_forward(detector, frame_bgr)
    enroll_detections = decode_enroll_face_detections(
        head_outputs=head_outputs,
        frame_width=frame_w,
        frame_height=frame_h,
        input_size=input_size,
        min_score=ENROLL_CANDIDATE_SCORE_FLOOR,
        nms_threshold=float(detector.get("nms_threshold", 0.45)),
    )
    if enroll_detections:
        return enroll_detections

    ncnn_state = detector.get("ncnn") or {}
    if ncnn_state.get("enroll_enabled"):
        return []

    cascade = detector.get("face_cascade")
    if cascade is None:
        return []

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
            area_ratio = (fw * fh) / float(max(1, frame_h * frame_w))
            heuristic_conf = max(0.55, min(0.92, 0.55 + (area_ratio * 80.0)))
            model_conf = _extract_ncnn_face_confidence(
                detector, frame_bgr, (fx, fy, fw, fh)
            )
            score = combine_smartface_face_score(
                heuristic_conf=heuristic_conf,
                model_conf=model_conf,
            )
            detections.append(
                {
                    "class": "face",
                    "bbox": (fx, fy, fw, fh),
                    "score": float(max(0.0, min(0.99, score))),
                }
            )

    return dedupe_face_detections(detections, overlap_threshold=0.45)


def filter_detections_by_score(
    detections: list[dict[str, Any]],
    min_score: float,
) -> list[dict[str, Any]]:
    threshold = float(max(0.0, min(1.0, min_score)))
    filtered: list[dict[str, Any]] = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue

        score = detection.get("score")
        if not isinstance(score, (int, float)):
            continue

        numeric = float(score)
        if not math.isfinite(numeric):
            continue
        if numeric < threshold:
            continue

        filtered.append(detection)

    return filtered


def run_smartface_detector(
    detector: dict[str, Any],
    frame_bgr: Any,
    candidate_boxes: list[Box],
) -> list[dict[str, Any]]:
    scored = score_smartface_face_candidates(detector, frame_bgr, candidate_boxes)
    min_score = float(detector.get("min_score", 0.75))
    return filter_detections_by_score(scored, min_score=min_score)


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
    smartface_lmk_param = config.get("smartface_lmk_param")
    smartface_lmk_bin = config.get("smartface_lmk_bin")
    smartface_extract_param = config.get("smartface_extract_param")
    smartface_extract_bin = config.get("smartface_extract_bin")
    smartface_identity_distance_threshold = float(
        config.get("smartface_identity_distance_threshold", 0.35)
    )
    smartface_identity_stable_frames = max(
        1,
        int(config.get("smartface_identity_stable_frames", 2)),
    )
    smartface_identity_min_face_score = float(
        config.get("smartface_identity_min_face_score", 0.75)
    )
    smartface_identity_min_face_area_ratio = float(
        config.get("smartface_identity_min_face_area_ratio", 0.0012)
    )
    smartface_identity_require_landmarks_raw = config.get(
        "smartface_identity_require_landmarks",
        False,
    )
    if isinstance(smartface_identity_require_landmarks_raw, bool):
        smartface_identity_require_landmarks = smartface_identity_require_landmarks_raw
    elif isinstance(smartface_identity_require_landmarks_raw, str):
        smartface_identity_require_landmarks = (
            smartface_identity_require_landmarks_raw.strip().lower()
            in {"1", "true", "yes", "on"}
        )
    else:
        smartface_identity_require_landmarks = bool(
            smartface_identity_require_landmarks_raw
        )
    smartface_identity_split_guard_ratio = float(
        config.get(
            "smartface_identity_split_guard_ratio",
            DEFAULT_IDENTITY_SPLIT_GUARD_RATIO,
        )
    )
    smartface_identity_split_guard_max_seen = max(
        0,
        int(
            config.get(
                "smartface_identity_split_guard_max_seen",
                DEFAULT_IDENTITY_SPLIT_GUARD_MAX_SEEN,
            )
        ),
    )
    smartface_identity_merge_recover_threshold = float(
        config.get(
            "smartface_identity_merge_recover_threshold",
            DEFAULT_IDENTITY_MERGE_RECOVER_THRESHOLD,
        )
    )
    smartface_identity_merge_recover_min_seen = max(
        1,
        int(
            config.get(
                "smartface_identity_merge_recover_min_seen",
                DEFAULT_IDENTITY_MERGE_RECOVER_MIN_SEEN,
            )
        ),
    )
    smartface_identity_prevent_duplicate_per_frame_raw = config.get(
        "smartface_identity_prevent_duplicate_per_frame",
        True,
    )
    if isinstance(smartface_identity_prevent_duplicate_per_frame_raw, bool):
        smartface_identity_prevent_duplicate_per_frame = (
            smartface_identity_prevent_duplicate_per_frame_raw
        )
    elif isinstance(smartface_identity_prevent_duplicate_per_frame_raw, str):
        smartface_identity_prevent_duplicate_per_frame = (
            smartface_identity_prevent_duplicate_per_frame_raw.strip().lower()
            in {"1", "true", "yes", "on"}
        )
    else:
        smartface_identity_prevent_duplicate_per_frame = bool(
            smartface_identity_prevent_duplicate_per_frame_raw
        )
    smartface_identity_verified_allowlist_mode_raw = config.get(
        "smartface_identity_verified_allowlist_mode",
        False,
    )
    if isinstance(smartface_identity_verified_allowlist_mode_raw, bool):
        smartface_identity_verified_allowlist_mode = (
            smartface_identity_verified_allowlist_mode_raw
        )
    elif isinstance(smartface_identity_verified_allowlist_mode_raw, str):
        smartface_identity_verified_allowlist_mode = (
            smartface_identity_verified_allowlist_mode_raw.strip().lower()
            in {"1", "true", "yes", "on"}
        )
    else:
        smartface_identity_verified_allowlist_mode = bool(
            smartface_identity_verified_allowlist_mode_raw
        )
    smartface_identity_quality_verifier_enabled_raw = config.get(
        "smartface_identity_quality_verifier_enabled",
        DEFAULT_IDENTITY_QUALITY_VERIFIER_ENABLED,
    )
    if isinstance(smartface_identity_quality_verifier_enabled_raw, bool):
        smartface_identity_quality_verifier_enabled = (
            smartface_identity_quality_verifier_enabled_raw
        )
    elif isinstance(smartface_identity_quality_verifier_enabled_raw, str):
        smartface_identity_quality_verifier_enabled = (
            smartface_identity_quality_verifier_enabled_raw.strip().lower()
            in {"1", "true", "yes", "on"}
        )
    else:
        smartface_identity_quality_verifier_enabled = bool(
            smartface_identity_quality_verifier_enabled_raw
        )
    smartface_identity_pose_mask_gate_enabled_raw = config.get(
        "smartface_identity_pose_mask_gate_enabled",
        DEFAULT_IDENTITY_POSE_MASK_GATE_ENABLED,
    )
    if isinstance(smartface_identity_pose_mask_gate_enabled_raw, bool):
        smartface_identity_pose_mask_gate_enabled = (
            smartface_identity_pose_mask_gate_enabled_raw
        )
    elif isinstance(smartface_identity_pose_mask_gate_enabled_raw, str):
        smartface_identity_pose_mask_gate_enabled = (
            smartface_identity_pose_mask_gate_enabled_raw.strip().lower()
            in {"1", "true", "yes", "on"}
        )
    else:
        smartface_identity_pose_mask_gate_enabled = bool(
            smartface_identity_pose_mask_gate_enabled_raw
        )
    smartface_identity_max_abs_yaw_deg = max(
        0.0,
        float(
            config.get(
                "smartface_identity_max_abs_yaw_deg",
                DEFAULT_IDENTITY_MAX_ABS_YAW_DEG,
            )
        ),
    )
    smartface_identity_max_abs_pitch_deg = max(
        0.0,
        float(
            config.get(
                "smartface_identity_max_abs_pitch_deg",
                DEFAULT_IDENTITY_MAX_ABS_PITCH_DEG,
            )
        ),
    )
    smartface_identity_max_abs_roll_deg = max(
        0.0,
        float(
            config.get(
                "smartface_identity_max_abs_roll_deg",
                DEFAULT_IDENTITY_MAX_ABS_ROLL_DEG,
            )
        ),
    )
    smartface_identity_mask_confidence_min = max(
        0.0,
        min(
            1.0,
            float(
                config.get(
                    "smartface_identity_mask_confidence_min",
                    DEFAULT_IDENTITY_MASK_CONFIDENCE_MIN,
                )
            ),
        ),
    )
    identity_gallery_path = config.get("identity_gallery_path")
    identity_gallery_max_profiles = max(
        1,
        int(
            config.get(
                "identity_gallery_max_profiles",
                DEFAULT_IDENTITY_GALLERY_MAX_PROFILES,
            )
        ),
    )
    identity_gallery_save_interval_frames = max(
        1,
        int(
            config.get(
                "identity_gallery_save_interval_frames",
                DEFAULT_IDENTITY_GALLERY_SAVE_INTERVAL_FRAMES,
            )
        ),
    )
    identity_gallery_max_idle_ms = max(
        0,
        int(
            config.get(
                "identity_gallery_max_idle_ms",
                DEFAULT_IDENTITY_GALLERY_MAX_IDLE_MS,
            )
        ),
    )
    identity_gallery_prune_interval_frames = max(
        1,
        int(
            config.get(
                "identity_gallery_prune_interval_frames",
                DEFAULT_IDENTITY_GALLERY_PRUNE_INTERVAL_FRAMES,
            )
        ),
    )

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
        smartface_lmk_param=(str(smartface_lmk_param) if smartface_lmk_param else None),
        smartface_lmk_bin=(str(smartface_lmk_bin) if smartface_lmk_bin else None),
        smartface_extract_param=(
            str(smartface_extract_param) if smartface_extract_param else None
        ),
        smartface_extract_bin=(
            str(smartface_extract_bin) if smartface_extract_bin else None
        ),
        smartface_identity_distance_threshold=smartface_identity_distance_threshold,
        smartface_identity_stable_frames=smartface_identity_stable_frames,
        smartface_identity_min_face_score=smartface_identity_min_face_score,
        smartface_identity_min_face_area_ratio=smartface_identity_min_face_area_ratio,
        smartface_identity_require_landmarks=smartface_identity_require_landmarks,
        smartface_identity_split_guard_ratio=smartface_identity_split_guard_ratio,
        smartface_identity_split_guard_max_seen=smartface_identity_split_guard_max_seen,
        smartface_identity_merge_recover_threshold=smartface_identity_merge_recover_threshold,
        smartface_identity_merge_recover_min_seen=smartface_identity_merge_recover_min_seen,
        smartface_identity_prevent_duplicate_per_frame=smartface_identity_prevent_duplicate_per_frame,
        smartface_identity_verified_allowlist_mode=smartface_identity_verified_allowlist_mode,
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
    prev_identity_ids: set[str] = set()
    event_counter = 0
    face_streak = 0
    identity_streaks: dict[str, int] = {}
    identity_state: dict[str, Any] = load_identity_gallery(
        str(identity_gallery_path) if identity_gallery_path else None,
        max_profiles=identity_gallery_max_profiles,
    )
    identity_state_dirty = False
    identity_decision_cumulative = new_identity_decision_counters()
    identity_quality_stats_cumulative = new_identity_quality_counters()

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
            face_score_source: list[dict[str, Any]] = []
            raw_face_candidate_count = 0
            face_threshold = None
            identity_candidates: dict[str, dict[str, float | None]] = {}
            stable_identity_ids: set[str] = set()
            identity_distance_threshold: float | None = None
            identity_stable_frames_threshold: int | None = None
            identity_min_face_score: float | None = None
            identity_min_face_area_ratio: float | None = None
            identity_require_landmarks: bool | None = None
            identity_rejected_count = 0
            identity_reject_reasons: dict[str, int] = {}
            identity_decision_frame = new_identity_decision_counters()
            identity_quality_stats_frame = new_identity_quality_counters()
            now_ms = int(time.time() * 1000)

            if (
                identity_gallery_max_idle_ms > 0
                and (idx % identity_gallery_prune_interval_frames) == 0
            ):
                removed_ids = prune_identity_gallery(
                    identity_state=identity_state,
                    now_ms=now_ms,
                    max_idle_ms=identity_gallery_max_idle_ms,
                )
                if removed_ids:
                    identity_state_dirty = True
                    for identity_id in removed_ids:
                        prev_identity_ids.discard(identity_id)
                        identity_streaks.pop(identity_id, None)

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
                scored_face_candidates = score_smartface_face_candidates(
                    detector,
                    frame,
                    face_candidates,
                )
                face_threshold = float(detector.get("min_score", smartface_min_score))
                face_dets = filter_detections_by_score(
                    scored_face_candidates,
                    min_score=face_threshold,
                )
                face_score_source = scored_face_candidates
                raw_face_candidate_count = len(scored_face_candidates)
                raw_face_count = len(face_dets)
                face_streak = face_streak + 1 if face_dets else 0
                stable_face_dets = (
                    face_dets if face_streak >= smartface_stable_frames else []
                )
                all_dets = person_dets + vehicle_dets + stable_face_dets

                identity_distance_threshold = float(
                    detector.get("identity_distance_threshold", 0.35)
                )
                identity_stable_frames_threshold = int(
                    detector.get("identity_stable_frames", 2)
                )
                identity_min_face_score = float(
                    detector.get("identity_min_face_score", face_threshold)
                )
                identity_min_face_area_ratio = float(
                    detector.get("identity_min_face_area_ratio", 0.0012)
                )
                identity_require_landmarks = bool(
                    detector.get("identity_require_landmarks", False)
                )
                identity_seen_this_frame: set[str] = set()
                for face_det in stable_face_dets:
                    bbox = face_det.get("bbox")
                    if not isinstance(bbox, tuple) or len(bbox) != 4:
                        identity_rejected_count += 1
                        identity_reject_reasons["invalid_bbox"] = (
                            int(identity_reject_reasons.get("invalid_bbox", 0)) + 1
                        )
                        continue

                    embedding, _landmarks = extract_face_embedding(
                        detector, frame, bbox
                    )
                    if not isinstance(embedding, list):
                        identity_rejected_count += 1
                        identity_reject_reasons["missing_embedding"] = (
                            int(identity_reject_reasons.get("missing_embedding", 0)) + 1
                        )
                        continue

                    accepted_quality, reject_reason = (
                        evaluate_identity_candidate_quality(
                            face_detection=face_det,
                            frame_area=frame_area,
                            landmarks=_landmarks,
                            min_face_score=float(identity_min_face_score),
                            min_face_area_ratio=float(identity_min_face_area_ratio),
                            require_landmarks=bool(identity_require_landmarks),
                        )
                    )
                    if not accepted_quality:
                        identity_rejected_count += 1
                        if isinstance(reject_reason, str) and reject_reason:
                            identity_reject_reasons[reject_reason] = (
                                int(identity_reject_reasons.get(reject_reason, 0)) + 1
                            )
                        continue

                    accepted_pose_mask, _reason, _details = (
                        apply_pose_mask_quality_verifier(
                            face_detection=face_det,
                            verifier_enabled=smartface_identity_quality_verifier_enabled,
                            hard_gate_enabled=smartface_identity_pose_mask_gate_enabled,
                            max_abs_yaw=smartface_identity_max_abs_yaw_deg,
                            max_abs_pitch=smartface_identity_max_abs_pitch_deg,
                            max_abs_roll=smartface_identity_max_abs_roll_deg,
                            mask_confidence_min=smartface_identity_mask_confidence_min,
                            quality_stats_frame=identity_quality_stats_frame,
                            quality_stats_cumulative=identity_quality_stats_cumulative,
                            identity_reject_reasons=identity_reject_reasons,
                        )
                    )
                    if not accepted_pose_mask:
                        identity_rejected_count += 1
                        continue

                    decision_audit: dict[str, Any] = {}
                    identity_id, distance, _is_new = match_embedding_identity(
                        identity_state=identity_state,
                        embedding=embedding,
                        distance_threshold=identity_distance_threshold,
                        timestamp_ms=now_ms,
                        split_guard_ratio=float(
                            detector.get(
                                "identity_split_guard_ratio",
                                DEFAULT_IDENTITY_SPLIT_GUARD_RATIO,
                            )
                        ),
                        split_guard_max_seen=int(
                            detector.get(
                                "identity_split_guard_max_seen",
                                DEFAULT_IDENTITY_SPLIT_GUARD_MAX_SEEN,
                            )
                        ),
                        merge_recover_threshold=float(
                            detector.get(
                                "identity_merge_recover_threshold",
                                DEFAULT_IDENTITY_MERGE_RECOVER_THRESHOLD,
                            )
                        ),
                        merge_recover_min_seen=int(
                            detector.get(
                                "identity_merge_recover_min_seen",
                                DEFAULT_IDENTITY_MERGE_RECOVER_MIN_SEEN,
                            )
                        ),
                        reserved_identity_ids=(
                            set(identity_seen_this_frame)
                            if bool(
                                detector.get(
                                    "identity_prevent_duplicate_per_frame",
                                    True,
                                )
                            )
                            else None
                        ),
                        verified_allowlist_mode=bool(
                            detector.get("identity_verified_allowlist_mode", False)
                        ),
                        audit=decision_audit,
                    )
                    bump_identity_decision_counter(
                        identity_decision_frame,
                        (
                            str(decision_audit.get("decision"))
                            if isinstance(decision_audit.get("decision"), str)
                            else None
                        ),
                    )
                    bump_identity_decision_counter(
                        identity_decision_cumulative,
                        (
                            str(decision_audit.get("decision"))
                            if isinstance(decision_audit.get("decision"), str)
                            else None
                        ),
                    )
                    identity_state_dirty = True
                    face_det["identityId"] = identity_id
                    if isinstance(distance, (int, float)) and math.isfinite(
                        float(distance)
                    ):
                        face_det["identityDistance"] = float(distance)

                    confidence = float(face_det.get("score", 0.5))
                    existing = identity_candidates.get(identity_id)
                    existing_conf = 0.0
                    if isinstance(existing, dict):
                        existing_value = existing.get("confidence")
                        if isinstance(existing_value, (int, float)) and math.isfinite(
                            float(existing_value)
                        ):
                            existing_conf = float(existing_value)
                    if existing is None or confidence > existing_conf:
                        identity_candidates[identity_id] = {
                            "confidence": confidence,
                            "distance": (
                                float(distance)
                                if isinstance(distance, (int, float))
                                and math.isfinite(float(distance))
                                else None
                            ),
                        }
                    identity_seen_this_frame.add(identity_id)

                for identity_id in list(identity_streaks.keys()):
                    if identity_id not in identity_seen_this_frame:
                        identity_streaks.pop(identity_id, None)

                for identity_id in identity_seen_this_frame:
                    streak = int(identity_streaks.get(identity_id, 0)) + 1
                    identity_streaks[identity_id] = streak
                    if streak >= identity_stable_frames_threshold:
                        stable_identity_ids.add(identity_id)

                identity_enters, identity_leaves = class_edges(
                    prev_identity_ids,
                    stable_identity_ids,
                )
                for identity_id in identity_enters:
                    event_counter += 1
                    details = identity_candidates.get(identity_id, {})
                    detail_conf = details.get("confidence")
                    confidence_value = (
                        float(detail_conf)
                        if isinstance(detail_conf, (int, float))
                        and math.isfinite(float(detail_conf))
                        else 0.5
                    )
                    detail_distance = details.get("distance")
                    distance_value = (
                        float(detail_distance)
                        if isinstance(detail_distance, (int, float))
                        and math.isfinite(float(detail_distance))
                        else None
                    )
                    emit_json(
                        make_identity_event_payload(
                            edge_type="enter",
                            event_id=f"rtsp-id-{event_counter}",
                            identity_id=identity_id,
                            confidence=confidence_value,
                            timestamp_ms=now_ms,
                            distance=distance_value,
                        )
                    )

                for identity_id in identity_leaves:
                    event_counter += 1
                    emit_json(
                        make_identity_event_payload(
                            edge_type="leave",
                            event_id=f"rtsp-id-{event_counter}",
                            identity_id=identity_id,
                            confidence=0.5,
                            timestamp_ms=now_ms,
                            distance=None,
                        )
                    )
                prev_identity_ids = set(stable_identity_ids)
            else:
                all_dets = run_model_detector(detector, frame)
                person_dets = [d for d in all_dets if d["class"] == "person"]
                vehicle_dets = [d for d in all_dets if d["class"] == "vehicle"]
                face_dets = [d for d in all_dets if d["class"] == "face"]
                raw_face_count = len(face_dets)
                face_score_source = face_dets
                raw_face_candidate_count = raw_face_count

            stable_face_count = (
                len(stable_face_dets)
                if detector.get("name") == "smartface_ncnn"
                else raw_face_count
            )
            face_score_telemetry = build_face_score_telemetry(
                class_detection_scores(face_score_source, "face")
            )
            face_acceptance_rate = (
                float(raw_face_count / raw_face_candidate_count)
                if raw_face_candidate_count > 0
                else None
            )

            curr_classes = {
                str(d.get("class"))
                for d in all_dets
                if isinstance(d, dict) and isinstance(d.get("class"), str)
            }
            enters, leaves = class_edges(prev_classes, curr_classes)
            prev_classes = set(curr_classes)

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

            identity_quality_summary_payload = build_identity_quality_summary_payload(
                verifier_enabled=smartface_identity_quality_verifier_enabled,
                hard_gate_enabled=smartface_identity_pose_mask_gate_enabled,
                quality_stats_frame=identity_quality_stats_frame,
                quality_stats_cumulative=identity_quality_stats_cumulative,
                max_abs_yaw=smartface_identity_max_abs_yaw_deg,
                max_abs_pitch=smartface_identity_max_abs_pitch_deg,
                max_abs_roll=smartface_identity_max_abs_roll_deg,
                mask_confidence_min=smartface_identity_mask_confidence_min,
            )

            heartbeat = {
                "functionName": "ParityFrameSummary",
                "payload": {
                    "frameIndex": idx,
                    "timestamp": now_ms,
                    "classes": sorted(curr_classes),
                    "personCount": len(person_dets),
                    "vehicleCount": len(vehicle_dets),
                    "faceCount": raw_face_count,
                    "rawFaceCandidateCount": raw_face_candidate_count,
                    "faceAcceptanceRate": face_acceptance_rate,
                    "stableFaceCount": stable_face_count,
                    "faceStreak": face_streak,
                    "smartfaceMinScoreThreshold": face_threshold,
                    "smartfaceStableFramesThreshold": (
                        smartface_stable_frames
                        if detector.get("name") == "smartface_ncnn"
                        else None
                    ),
                    "identityCount": len(stable_identity_ids),
                    "identityIds": sorted(stable_identity_ids),
                    "identityCandidateCount": len(identity_candidates),
                    "identityProfileCount": len(identity_state.get("profiles", {})),
                    "identityDistanceThreshold": identity_distance_threshold,
                    "identityStableFramesThreshold": identity_stable_frames_threshold,
                    "identityMinFaceScore": identity_min_face_score,
                    "identityMinFaceAreaRatio": identity_min_face_area_ratio,
                    "identityRequireLandmarks": identity_require_landmarks,
                    "identitySplitGuardRatio": detector.get(
                        "identity_split_guard_ratio"
                    ),
                    "identitySplitGuardMaxSeen": detector.get(
                        "identity_split_guard_max_seen"
                    ),
                    "identityMergeRecoverThreshold": detector.get(
                        "identity_merge_recover_threshold"
                    ),
                    "identityMergeRecoverMinSeen": detector.get(
                        "identity_merge_recover_min_seen"
                    ),
                    "identityPreventDuplicatePerFrame": detector.get(
                        "identity_prevent_duplicate_per_frame"
                    ),
                    "identityVerifiedAllowlistMode": detector.get(
                        "identity_verified_allowlist_mode"
                    ),
                    "identityVerifiedAllowlistSize": len(
                        _as_string_set(
                            _normalize_uiface_policy(
                                identity_state.get("uifacePolicy")
                            ).get("verifiedUniqueIds")
                        )
                    ),
                    "identityRejectedCount": identity_rejected_count,
                    "identityRejectReasons": identity_reject_reasons,
                    "identityDecisionAuditFrame": identity_decision_frame,
                    "identityDecisionAuditCumulative": identity_decision_cumulative,
                    **identity_quality_summary_payload,
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

            if (
                identity_state_dirty
                and identity_gallery_path
                and ((idx + 1) % identity_gallery_save_interval_frames == 0)
            ):
                if save_identity_gallery(
                    str(identity_gallery_path),
                    identity_state,
                    max_profiles=identity_gallery_max_profiles,
                ):
                    identity_state_dirty = False

            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

            idx += 1
    finally:
        cap.release()
        if identity_state_dirty and identity_gallery_path:
            save_identity_gallery(
                str(identity_gallery_path),
                identity_state,
                max_profiles=identity_gallery_max_profiles,
            )

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
        "--smartface-lmk-param",
        default=None,
        help="Optional path to SmartFace landmark NCNN .param file",
    )
    parser.add_argument(
        "--smartface-lmk-bin",
        default=None,
        help="Optional path to SmartFace landmark NCNN .bin file",
    )
    parser.add_argument(
        "--smartface-extract-param",
        default=None,
        help="Optional path to SmartFace extract NCNN .param file",
    )
    parser.add_argument(
        "--smartface-extract-bin",
        default=None,
        help="Optional path to SmartFace extract NCNN .bin file",
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
    parser.add_argument(
        "--smartface-identity-distance-threshold",
        type=float,
        default=0.35,
        help="Maximum cosine distance for reusing an identity profile",
    )
    parser.add_argument(
        "--smartface-identity-stable-frames",
        type=int,
        default=2,
        help="Consecutive frames required before emitting identity transitions",
    )
    parser.add_argument(
        "--smartface-identity-min-face-score",
        type=float,
        default=0.75,
        help="Minimum face score required before identity matching",
    )
    parser.add_argument(
        "--smartface-identity-min-face-area-ratio",
        type=float,
        default=0.0012,
        help="Minimum bbox area ratio required before identity matching",
    )
    parser.add_argument(
        "--smartface-identity-require-landmarks",
        action="store_true",
        help="Require valid landmarks before identity matching",
    )
    parser.add_argument(
        "--smartface-identity-split-guard-ratio",
        type=float,
        default=DEFAULT_IDENTITY_SPLIT_GUARD_RATIO,
        help="Near-threshold ratio that triggers split correction for low-seen profiles",
    )
    parser.add_argument(
        "--smartface-identity-split-guard-max-seen",
        type=int,
        default=DEFAULT_IDENTITY_SPLIT_GUARD_MAX_SEEN,
        help="Maximum seen-count considered low confidence for split guard",
    )
    parser.add_argument(
        "--smartface-identity-merge-recover-threshold",
        type=float,
        default=DEFAULT_IDENTITY_MERGE_RECOVER_THRESHOLD,
        help="Upper distance threshold used to recover accidental identity splits",
    )
    parser.add_argument(
        "--smartface-identity-merge-recover-min-seen",
        type=int,
        default=DEFAULT_IDENTITY_MERGE_RECOVER_MIN_SEEN,
        help="Minimum profile seen-count required for merge-recovery reuse",
    )
    parser.add_argument(
        "--smartface-identity-allow-duplicate-per-frame",
        dest="smartface_identity_prevent_duplicate_per_frame",
        action="store_false",
        help="Allow assigning one identity to multiple faces in the same frame",
    )
    parser.set_defaults(smartface_identity_prevent_duplicate_per_frame=True)
    parser.add_argument(
        "--smartface-identity-verified-allowlist-mode",
        action="store_true",
        help="Restrict identity profile matching to uiface verified unique IDs when present",
    )
    parser.add_argument(
        "--smartface-identity-quality-verifier-enabled",
        action="store_true",
        help="Enable pose/mask quality verifier before identity matching",
    )
    parser.add_argument(
        "--smartface-identity-pose-mask-gate-enabled",
        action="store_true",
        help="Hard-gate identity matching when pose/mask quality is out of policy",
    )
    parser.add_argument(
        "--smartface-identity-max-abs-yaw-deg",
        type=float,
        default=None,
        help="Maximum allowed absolute yaw angle before gating",
    )
    parser.add_argument(
        "--smartface-identity-max-abs-pitch-deg",
        type=float,
        default=None,
        help="Maximum allowed absolute pitch angle before gating",
    )
    parser.add_argument(
        "--smartface-identity-max-abs-roll-deg",
        type=float,
        default=None,
        help="Maximum allowed absolute roll angle before gating",
    )
    parser.add_argument(
        "--smartface-identity-mask-confidence-min",
        type=float,
        default=None,
        help="Minimum mask confidence allowed before gating",
    )
    parser.add_argument(
        "--identity-gallery-path",
        default=None,
        help="Optional JSON path for persisted identity gallery",
    )
    parser.add_argument(
        "--identity-gallery-max-profiles",
        type=int,
        default=DEFAULT_IDENTITY_GALLERY_MAX_PROFILES,
        help="Maximum identity profiles kept in persisted gallery",
    )
    parser.add_argument(
        "--identity-gallery-save-interval-frames",
        type=int,
        default=DEFAULT_IDENTITY_GALLERY_SAVE_INTERVAL_FRAMES,
        help="Frame interval between persisted identity gallery saves",
    )
    parser.add_argument(
        "--identity-gallery-max-idle-ms",
        type=int,
        default=DEFAULT_IDENTITY_GALLERY_MAX_IDLE_MS,
        help="Maximum idle time before a profile is evicted",
    )
    parser.add_argument(
        "--identity-gallery-prune-interval-frames",
        type=int,
        default=DEFAULT_IDENTITY_GALLERY_PRUNE_INTERVAL_FRAMES,
        help="Frame interval between stale-profile cleanup passes",
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
        "smartface_lmk_param": args.smartface_lmk_param,
        "smartface_lmk_bin": args.smartface_lmk_bin,
        "smartface_extract_param": args.smartface_extract_param,
        "smartface_extract_bin": args.smartface_extract_bin,
        "smartface_input_size": max(16, int(args.smartface_input_size)),
        "smartface_min_score": max(0.0, float(args.smartface_min_score)),
        "smartface_stable_frames": max(1, int(args.smartface_stable_frames)),
        "smartface_identity_distance_threshold": max(
            0.0,
            min(2.0, float(args.smartface_identity_distance_threshold)),
        ),
        "smartface_identity_stable_frames": max(
            1,
            int(args.smartface_identity_stable_frames),
        ),
        "smartface_identity_min_face_score": max(
            0.0,
            min(1.0, float(args.smartface_identity_min_face_score)),
        ),
        "smartface_identity_min_face_area_ratio": max(
            0.0,
            min(1.0, float(args.smartface_identity_min_face_area_ratio)),
        ),
        "smartface_identity_require_landmarks": bool(
            args.smartface_identity_require_landmarks
        ),
        "smartface_identity_split_guard_ratio": max(
            0.0,
            min(1.0, float(args.smartface_identity_split_guard_ratio)),
        ),
        "smartface_identity_split_guard_max_seen": max(
            0,
            int(args.smartface_identity_split_guard_max_seen),
        ),
        "smartface_identity_merge_recover_threshold": max(
            0.0,
            min(2.0, float(args.smartface_identity_merge_recover_threshold)),
        ),
        "smartface_identity_merge_recover_min_seen": max(
            1,
            int(args.smartface_identity_merge_recover_min_seen),
        ),
        "smartface_identity_prevent_duplicate_per_frame": bool(
            args.smartface_identity_prevent_duplicate_per_frame
        ),
        "smartface_identity_verified_allowlist_mode": bool(
            args.smartface_identity_verified_allowlist_mode
        ),
        "identity_gallery_path": args.identity_gallery_path,
        "identity_gallery_max_profiles": max(
            1,
            int(args.identity_gallery_max_profiles),
        ),
        "identity_gallery_save_interval_frames": max(
            1,
            int(args.identity_gallery_save_interval_frames),
        ),
        "identity_gallery_max_idle_ms": max(
            0,
            int(args.identity_gallery_max_idle_ms),
        ),
        "identity_gallery_prune_interval_frames": max(
            1,
            int(args.identity_gallery_prune_interval_frames),
        ),
    }
    if bool(args.smartface_identity_quality_verifier_enabled):
        cfg["smartface_identity_quality_verifier_enabled"] = True
    if bool(args.smartface_identity_pose_mask_gate_enabled):
        cfg["smartface_identity_pose_mask_gate_enabled"] = True
    if args.smartface_identity_max_abs_yaw_deg is not None:
        cfg["smartface_identity_max_abs_yaw_deg"] = max(
            0.0,
            float(args.smartface_identity_max_abs_yaw_deg),
        )
    if args.smartface_identity_max_abs_pitch_deg is not None:
        cfg["smartface_identity_max_abs_pitch_deg"] = max(
            0.0,
            float(args.smartface_identity_max_abs_pitch_deg),
        )
    if args.smartface_identity_max_abs_roll_deg is not None:
        cfg["smartface_identity_max_abs_roll_deg"] = max(
            0.0,
            float(args.smartface_identity_max_abs_roll_deg),
        )
    if args.smartface_identity_mask_confidence_min is not None:
        cfg["smartface_identity_mask_confidence_min"] = max(
            0.0,
            min(1.0, float(args.smartface_identity_mask_confidence_min)),
        )
    return run_parity(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
