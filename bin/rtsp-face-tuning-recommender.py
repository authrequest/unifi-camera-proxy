#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_JSONL = (
    REPO_ROOT / "analysis/harness/reports/semantic_validation/rtsp_parity_events.jsonl"
)
DEFAULT_OUTPUT_REPORT = (
    REPO_ROOT
    / "analysis/harness/reports/semantic_validation/rtsp_face_tuning_recommendation.json"
)
IDENTITY_AUDIT_KEYS = (
    "strict_match",
    "merge_recover",
    "new_identity_split_guard",
    "new_identity_reserved",
    "new_identity_threshold",
    "new_identity_no_profiles",
    "new_identity_invalid_embedding",
    "unknown",
)


def _as_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _as_non_negative_int(value: Any) -> int | None:
    if not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]

    position = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _iter_frame_payloads(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            if not isinstance(event, dict):
                continue
            if event.get("functionName") != "ParityFrameSummary":
                continue

            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            frames.append(payload)

    return frames


def analyze_parity_jsonl(
    input_jsonl: str | Path,
    *,
    low_acceptance_threshold: float = 0.35,
    high_acceptance_threshold: float = 0.90,
    low_stability_ratio: float = 0.35,
    high_stability_ratio: float = 0.95,
) -> dict[str, Any]:
    source_path = Path(input_jsonl).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"RTSP parity JSONL not found: {source_path}")

    frame_payloads = _iter_frame_payloads(source_path)
    if not frame_payloads:
        raise ValueError("No ParityFrameSummary payloads found in input JSONL")

    accepted_counts: list[int] = []
    raw_counts: list[int] = []
    stable_counts: list[int] = []
    acceptance_rates: list[float] = []
    p90_scores: list[float] = []
    min_score_thresholds: list[float] = []
    stable_frame_thresholds: list[int] = []
    split_guard_ratios: list[float] = []
    merge_recover_thresholds: list[float] = []
    face_streaks: list[int] = []
    stability_ratios: list[float] = []
    identity_audit_totals: dict[str, int] = {key: 0 for key in IDENTITY_AUDIT_KEYS}

    for payload in frame_payloads:
        face_count = _as_non_negative_int(payload.get("faceCount"))
        if face_count is None:
            face_count = 0

        raw_count = _as_non_negative_int(payload.get("rawFaceCandidateCount"))
        if raw_count is None:
            raw_count = face_count

        stable_count = _as_non_negative_int(payload.get("stableFaceCount"))
        if stable_count is None:
            stable_count = face_count

        face_streak = _as_non_negative_int(payload.get("faceStreak"))
        if face_streak is not None:
            face_streaks.append(face_streak)

        accepted_counts.append(face_count)
        raw_counts.append(raw_count)
        stable_counts.append(stable_count)

        threshold = _as_float(payload.get("smartfaceMinScoreThreshold"))
        if threshold is not None:
            min_score_thresholds.append(_clamp(threshold, 0.0, 1.0))

        split_guard_ratio = _as_float(payload.get("identitySplitGuardRatio"))
        if split_guard_ratio is not None:
            split_guard_ratios.append(_clamp(split_guard_ratio, 0.0, 1.0))

        merge_recover_threshold = _as_float(
            payload.get("identityMergeRecoverThreshold")
        )
        if merge_recover_threshold is not None:
            merge_recover_thresholds.append(_clamp(merge_recover_threshold, 0.0, 2.0))

        stable_threshold = _as_non_negative_int(
            payload.get("smartfaceStableFramesThreshold")
        )
        if stable_threshold is not None:
            stable_frame_thresholds.append(max(1, stable_threshold))

        acceptance_rate = _as_float(payload.get("faceAcceptanceRate"))
        if acceptance_rate is None and raw_count > 0:
            acceptance_rate = face_count / raw_count
        if acceptance_rate is not None:
            acceptance_rates.append(_clamp(acceptance_rate, 0.0, 1.0))

        if face_count > 0:
            stability_ratios.append(_clamp(stable_count / face_count, 0.0, 1.0))

        identity_decision_audit = payload.get("identityDecisionAuditFrame")
        if isinstance(identity_decision_audit, dict):
            for key in IDENTITY_AUDIT_KEYS:
                raw = identity_decision_audit.get(key)
                if not isinstance(raw, (int, float)):
                    continue
                numeric = float(raw)
                if not math.isfinite(numeric) or numeric < 0:
                    continue
                identity_audit_totals[key] = int(
                    identity_audit_totals[key] + int(numeric)
                )

        face_score_telemetry = payload.get("faceScoreTelemetry")
        if not isinstance(face_score_telemetry, dict):
            continue
        summary = face_score_telemetry.get("summary")
        if not isinstance(summary, dict):
            continue
        p90 = _as_float(summary.get("p90"))
        if p90 is not None:
            p90_scores.append(_clamp(p90, 0.0, 1.0))

    current_min_score = _quantile(min_score_thresholds, 0.5)
    if current_min_score is None:
        current_min_score = 0.75

    current_stable_frames_f = _quantile(
        [float(value) for value in stable_frame_thresholds], 0.5
    )
    current_stable_frames = (
        max(1, int(round(current_stable_frames_f)))
        if current_stable_frames_f is not None
        else 2
    )

    current_split_guard_ratio = _quantile(split_guard_ratios, 0.5)
    if current_split_guard_ratio is None:
        current_split_guard_ratio = 0.90

    current_merge_recover_threshold = _quantile(merge_recover_thresholds, 0.5)
    if current_merge_recover_threshold is None:
        current_merge_recover_threshold = 0.45

    median_acceptance = _quantile(acceptance_rates, 0.5)
    median_stability_ratio = _quantile(stability_ratios, 0.5)
    median_p90 = _quantile(p90_scores, 0.5)
    median_face_streak_f = _quantile([float(value) for value in face_streaks], 0.5)
    median_face_streak = (
        int(round(median_face_streak_f)) if median_face_streak_f is not None else 0
    )

    identity_decision_total = sum(identity_audit_totals.values())

    def _audit_rate(key: str) -> float | None:
        if identity_decision_total <= 0:
            return None
        return float(identity_audit_totals.get(key, 0) / identity_decision_total)

    strict_match_rate = _audit_rate("strict_match")
    merge_recover_rate = _audit_rate("merge_recover")
    split_guard_new_rate = _audit_rate("new_identity_split_guard")
    threshold_new_rate = _audit_rate("new_identity_threshold")

    recommended_min_score = current_min_score
    min_score_action = "keep"
    if median_acceptance is not None:
        if median_acceptance < low_acceptance_threshold:
            min_score_action = "decrease"
            lower_target = current_min_score - 0.03
            if median_p90 is not None:
                lower_target = min(lower_target, median_p90 - 0.02)
            recommended_min_score = _clamp(lower_target, 0.55, 0.95)
        elif median_acceptance > high_acceptance_threshold:
            min_score_action = "increase"
            upper_target = current_min_score + 0.03
            if median_p90 is not None:
                upper_target = max(upper_target, min(0.95, median_p90 - 0.01))
            recommended_min_score = _clamp(upper_target, 0.55, 0.95)

    recommended_stable_frames = current_stable_frames
    stable_frames_action = "keep"
    if median_stability_ratio is not None:
        if (
            median_stability_ratio < low_stability_ratio
            and median_face_streak < current_stable_frames
        ):
            stable_frames_action = "decrease"
            recommended_stable_frames = max(1, current_stable_frames - 1)
        elif (
            median_stability_ratio > high_stability_ratio
            and median_acceptance is not None
            and median_acceptance > high_acceptance_threshold
        ):
            stable_frames_action = "increase"
            recommended_stable_frames = current_stable_frames + 1

    recommended_split_guard_ratio = current_split_guard_ratio
    split_guard_action = "keep"
    if split_guard_new_rate is not None:
        if split_guard_new_rate >= 0.18:
            split_guard_action = "decrease"
            recommended_split_guard_ratio = _clamp(
                current_split_guard_ratio - 0.05,
                0.65,
                0.99,
            )
        elif (
            strict_match_rate is not None
            and strict_match_rate >= 0.85
            and threshold_new_rate is not None
            and threshold_new_rate <= 0.02
            and split_guard_new_rate <= 0.03
        ):
            split_guard_action = "increase"
            recommended_split_guard_ratio = _clamp(
                current_split_guard_ratio + 0.03,
                0.65,
                0.99,
            )

    recommended_merge_recover_threshold = current_merge_recover_threshold
    merge_recover_action = "keep"
    if threshold_new_rate is not None and merge_recover_rate is not None:
        if threshold_new_rate >= 0.20 and merge_recover_rate <= 0.05:
            merge_recover_action = "increase"
            recommended_merge_recover_threshold = _clamp(
                current_merge_recover_threshold + 0.03,
                0.30,
                0.90,
            )
        elif merge_recover_rate >= 0.35 and threshold_new_rate <= 0.05:
            merge_recover_action = "decrease"
            recommended_merge_recover_threshold = _clamp(
                current_merge_recover_threshold - 0.03,
                0.30,
                0.90,
            )

    rationale: list[str] = []
    if min_score_action == "decrease":
        rationale.append(
            "Face acceptance is below target; lowering min score should recover valid detections."
        )
    elif min_score_action == "increase":
        rationale.append(
            "Face acceptance is very high; raising min score should reduce low-value detections."
        )
    else:
        rationale.append(
            "Current min score appears balanced against observed acceptance."
        )

    if stable_frames_action == "decrease":
        rationale.append(
            "Stable face ratio is low with short streaks; reducing stable frames should reduce missed transitions."
        )
    elif stable_frames_action == "increase":
        rationale.append(
            "Stable ratio and acceptance are both high; increasing stable frames can suppress flicker noise."
        )
    else:
        rationale.append(
            "Current stable-frame gate appears consistent with observed streak behavior."
        )

    if split_guard_action == "decrease":
        rationale.append(
            "Split-guard new-ID decisions are frequent; lowering split guard should reduce unnecessary identity splits."
        )
    elif split_guard_action == "increase":
        rationale.append(
            "Strict matches dominate with few split events; increasing split guard can reduce accidental merges."
        )
    else:
        rationale.append(
            "Split guard appears balanced against observed strict/split decision mix."
        )

    if merge_recover_action == "increase":
        rationale.append(
            "Threshold-based new identities dominate while merge-recover is rare; increasing merge-recover threshold can reduce duplicate IDs."
        )
    elif merge_recover_action == "decrease":
        rationale.append(
            "Merge-recover decisions dominate; decreasing merge-recover threshold can tighten identity reuse."
        )
    else:
        rationale.append(
            "Merge-recover threshold appears aligned with observed recovery behavior."
        )

    frames_with_candidates = sum(1 for count in raw_counts if count > 0)

    return {
        "version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_jsonl": str(source_path),
        "frameSummary": {
            "frameCount": len(frame_payloads),
            "framesWithCandidates": frames_with_candidates,
            "meanRawFaceCandidateCount": (
                float(fmean(raw_counts)) if raw_counts else None
            ),
            "meanAcceptedFaceCount": (
                float(fmean(accepted_counts)) if accepted_counts else None
            ),
            "meanStableFaceCount": (
                float(fmean(stable_counts)) if stable_counts else None
            ),
        },
        "observations": {
            "medianAcceptanceRate": median_acceptance,
            "medianStabilityRatio": median_stability_ratio,
            "medianP90Score": median_p90,
            "medianFaceStreak": median_face_streak,
            "medianMinScoreThreshold": current_min_score,
            "medianStableFramesThreshold": current_stable_frames,
            "medianIdentitySplitGuardRatio": current_split_guard_ratio,
            "medianIdentityMergeRecoverThreshold": current_merge_recover_threshold,
            "identityDecisionTotal": identity_decision_total,
            "identityDecisionCounts": identity_audit_totals,
            "strictMatchRate": strict_match_rate,
            "mergeRecoverRate": merge_recover_rate,
            "splitGuardNewRate": split_guard_new_rate,
            "thresholdNewRate": threshold_new_rate,
        },
        "recommendation": {
            "smartfaceMinScore": round(float(recommended_min_score), 4),
            "smartfaceStableFrames": int(recommended_stable_frames),
            "minScoreAction": min_score_action,
            "stableFramesAction": stable_frames_action,
            "smartfaceIdentitySplitGuardRatio": round(
                float(recommended_split_guard_ratio),
                4,
            ),
            "smartfaceIdentityMergeRecoverThreshold": round(
                float(recommended_merge_recover_threshold),
                4,
            ),
            "splitGuardAction": split_guard_action,
            "mergeRecoverAction": merge_recover_action,
            "rationale": rationale,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze RTSP parity JSONL and recommend smartfaceMinScore/smartfaceStableFrames"
        )
    )
    parser.add_argument(
        "--input-jsonl",
        default=str(DEFAULT_INPUT_JSONL),
        help="Path to newline-delimited parity events JSON",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_REPORT),
        help="Path to write recommendation report JSON",
    )
    parser.add_argument(
        "--low-acceptance-threshold",
        type=float,
        default=0.35,
        help="Lower bound for healthy face acceptance rate",
    )
    parser.add_argument(
        "--high-acceptance-threshold",
        type=float,
        default=0.90,
        help="Upper bound for healthy face acceptance rate",
    )
    parser.add_argument(
        "--low-stability-ratio",
        type=float,
        default=0.35,
        help="Lower bound for stableFaceCount/faceCount ratio",
    )
    parser.add_argument(
        "--high-stability-ratio",
        type=float,
        default=0.95,
        help="Upper bound for stableFaceCount/faceCount ratio",
    )
    args = parser.parse_args()

    report = analyze_parity_jsonl(
        input_jsonl=args.input_jsonl,
        low_acceptance_threshold=float(args.low_acceptance_threshold),
        high_acceptance_threshold=float(args.high_acceptance_threshold),
        low_stability_ratio=float(args.low_stability_ratio),
        high_stability_ratio=float(args.high_stability_ratio),
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote face tuning recommendation report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
