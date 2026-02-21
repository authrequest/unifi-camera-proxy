#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from typing import Any


MARKER_PATTERN = re.compile(
    r"ncnn|mnn|onnx|trt|yolox|ipu|tensor|param|model", re.IGNORECASE
)

MODEL_ARTIFACT_SUFFIXES = {".img", ".bin", ".param"}

EXCLUDED_ARTIFACT_NAMES = {
    "ipu_firmware.bin",
    "compatibility-matrix.csv",
    "inventory.csv",
    "fingerprints.json",
    "probe-results.json",
}


def run_command(args: list[str]) -> dict[str, Any]:
    command_text = " ".join(args)
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, errors="replace", check=False
        )
        return {
            "command": command_text,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "error": None,
        }
    except FileNotFoundError:
        return {
            "command": command_text,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "error": f"missing executable: {args[0]}",
        }


def file_hash(path: pathlib.Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def should_include_artifact(path: pathlib.Path) -> tuple[bool, str]:
    if path.name in EXCLUDED_ARTIFACT_NAMES:
        return False, "excluded-by-name"
    if path.suffix.lower() not in MODEL_ARTIFACT_SUFFIXES:
        return False, "unsupported-extension"
    return True, "model-candidate"


def counterpart_exists(
    path: pathlib.Path, counterpart_suffix: str, file_set: set[pathlib.Path]
) -> bool:
    if path.suffix.lower() == counterpart_suffix:
        return False
    return path.with_suffix(counterpart_suffix) in file_set


def classify_artifact(
    path: pathlib.Path, file_set: set[pathlib.Path], marker_hits: list[str]
) -> dict[str, str]:
    suffix = path.suffix.lower()
    paired_bin = counterpart_exists(path, ".bin", file_set)
    paired_param = counterpart_exists(path, ".param", file_set)
    marker_blob = "\n".join(marker_hits)

    if suffix == ".img":
        return {
            "likely_loader_runtime": "SigmaStar MI_IPU path (likely)",
            "reuse_outlook": "Low",
            "classification": "vendor-bound-candidate",
            "primary_portability_risk": "Likely SoC/offline-compiled container tied to MI_IPU runtime",
            "required_validators": "file; xxd -g 1 -l 256; strings; binwalk signature scan",
            "confidence": "medium-high",
            "heuristic_basis": "img extension + SmartDetect_sstar/MI_IPU evidence",
        }

    if suffix == ".param":
        has_ncnn_markers = bool(
            re.search(r"ncnn|param", marker_blob, flags=re.IGNORECASE)
        )
        confidence = "medium" if paired_bin or has_ncnn_markers else "low-medium"
        return {
            "likely_loader_runtime": "NCNN-style graph/param sidecar (likely)",
            "reuse_outlook": "Medium-High" if paired_bin else "Medium",
            "classification": "conditionally-portable-candidate",
            "primary_portability_risk": "Requires matching weight blob and compatible preprocessing/custom ops",
            "required_validators": "pair with .bin; local ncnn open/load probe; compare tensor names",
            "confidence": confidence,
            "heuristic_basis": "param extension and backend marker scan",
        }

    if suffix == ".bin":
        has_model_markers = bool(
            re.search(r"model|tensor|yolox|ncnn|ipu", marker_blob, flags=re.IGNORECASE)
        )
        confidence = "medium" if paired_param or has_model_markers else "low-medium"
        return {
            "likely_loader_runtime": "NCNN-like weights/blob or vendor binary payload",
            "reuse_outlook": "Medium" if paired_param else "Low-Medium",
            "classification": "unknown-until-paired",
            "primary_portability_risk": "Ambiguous blob role without pairing/manifest",
            "required_validators": "check paired .param; run ncnn probe; inspect header/signatures",
            "confidence": confidence,
            "heuristic_basis": "bin extension with optional param pairing",
        }

    return {
        "likely_loader_runtime": "Unknown",
        "reuse_outlook": "Unknown",
        "classification": "unknown",
        "primary_portability_risk": "No strong extension or marker mapping",
        "required_validators": "file; strings; binwalk; manual mapping",
        "confidence": "low",
        "heuristic_basis": "fallback",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate inventory/fingerprint/compatibility reports for copied AI model artifacts."
    )
    parser.add_argument(
        "--input-dir",
        default="analysis/harness/input/ai_model",
        help="Directory containing copied model artifacts (default: analysis/harness/input/ai_model)",
    )
    parser.add_argument(
        "--report-dir",
        default="analysis/harness/reports",
        help="Directory to write generated reports (default: analysis/harness/reports)",
    )
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir).resolve()
    report_dir = pathlib.Path(args.report_dir).resolve()
    binwalk_report_dir = report_dir / "binwalk"

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    report_dir.mkdir(parents=True, exist_ok=True)
    binwalk_report_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted([p for p in input_dir.iterdir() if p.is_file()])
    files: list[pathlib.Path] = []
    excluded_artifacts: list[dict[str, str]] = []
    for path in all_files:
        include, reason = should_include_artifact(path)
        if include:
            files.append(path)
        else:
            excluded_artifacts.append({"artifact": str(path), "reason": reason})

    file_set = set(files)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

    inventory_rows: list[dict[str, Any]] = []
    fingerprint_entries: list[dict[str, Any]] = []
    compatibility_rows: list[dict[str, Any]] = []

    for artifact in files:
        stat = artifact.stat()
        sha256 = file_hash(artifact, "sha256")
        md5 = file_hash(artifact, "md5")

        file_info = run_command(["file", str(artifact)])
        xxd_info = run_command(["xxd", "-g", "1", "-l", "256", str(artifact)])
        strings_info = run_command(["strings", str(artifact)])
        binwalk_info = run_command(["binwalk", str(artifact)])

        marker_hits: list[str] = []
        if strings_info["exit_code"] == 0 and strings_info["stdout"]:
            for line in strings_info["stdout"].splitlines():
                if MARKER_PATTERN.search(line):
                    marker_hits.append(line)
                if len(marker_hits) >= 200:
                    break

        binwalk_report_path = binwalk_report_dir / f"{artifact.name}.binwalk.txt"
        with binwalk_report_path.open("w", encoding="utf-8") as bf:
            bf.write(f"command: {binwalk_info['command']}\n")
            bf.write(f"exit_code: {binwalk_info['exit_code']}\n")
            if binwalk_info["error"]:
                bf.write(f"error: {binwalk_info['error']}\n")
            if binwalk_info["stderr"]:
                bf.write("stderr:\n")
                bf.write(binwalk_info["stderr"])
                if not binwalk_info["stderr"].endswith("\n"):
                    bf.write("\n")
            bf.write("stdout:\n")
            bf.write(binwalk_info["stdout"])

        inventory_rows.append(
            {
                "artifact": str(artifact),
                "filename": artifact.name,
                "size_bytes": stat.st_size,
                "mtime": dt.datetime.fromtimestamp(
                    stat.st_mtime, tz=dt.timezone.utc
                ).isoformat(),
                "sha256": sha256,
                "md5": md5,
            }
        )

        classification = classify_artifact(artifact, file_set, marker_hits)
        paired_param_exists = counterpart_exists(artifact, ".param", file_set)
        paired_bin_exists = counterpart_exists(artifact, ".bin", file_set)

        compatibility_rows.append(
            {
                "artifact": str(artifact),
                "likely_loader_runtime": classification["likely_loader_runtime"],
                "reuse_outlook": classification["reuse_outlook"],
                "classification": classification["classification"],
                "primary_portability_risk": classification["primary_portability_risk"],
                "required_validators": classification["required_validators"],
                "confidence": classification["confidence"],
                "heuristic_basis": classification["heuristic_basis"],
                "paired_param_exists": str(paired_param_exists).lower(),
                "paired_bin_exists": str(paired_bin_exists).lower(),
            }
        )

        fingerprint_entries.append(
            {
                "artifact": str(artifact),
                "sha256": sha256,
                "md5": md5,
                "file": {
                    "exit_code": file_info["exit_code"],
                    "error": file_info["error"],
                    "stdout": file_info["stdout"].strip(),
                    "stderr": file_info["stderr"].strip(),
                },
                "xxd_256": {
                    "exit_code": xxd_info["exit_code"],
                    "error": xxd_info["error"],
                    "stdout": xxd_info["stdout"],
                    "stderr": xxd_info["stderr"].strip(),
                },
                "strings": {
                    "exit_code": strings_info["exit_code"],
                    "error": strings_info["error"],
                    "marker_pattern": MARKER_PATTERN.pattern,
                    "marker_hits": marker_hits,
                },
                "binwalk": {
                    "exit_code": binwalk_info["exit_code"],
                    "error": binwalk_info["error"],
                    "report": str(binwalk_report_path),
                },
            }
        )

    inventory_path = report_dir / "inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["artifact", "filename", "size_bytes", "mtime", "sha256", "md5"],
        )
        writer.writeheader()
        writer.writerows(inventory_rows)

    compatibility_path = report_dir / "compatibility-matrix.csv"
    with compatibility_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "artifact",
                "likely_loader_runtime",
                "reuse_outlook",
                "classification",
                "primary_portability_risk",
                "required_validators",
                "confidence",
                "heuristic_basis",
                "paired_param_exists",
                "paired_bin_exists",
            ],
        )
        writer.writeheader()
        writer.writerows(compatibility_rows)

    fingerprints_path = report_dir / "fingerprints.json"
    with fingerprints_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": generated_at,
                "input_dir": str(input_dir),
                "artifact_count": len(fingerprint_entries),
                "excluded_count": len(excluded_artifacts),
                "excluded_artifacts": excluded_artifacts,
                "marker_pattern": MARKER_PATTERN.pattern,
                "artifacts": fingerprint_entries,
            },
            f,
            indent=2,
        )

    probe_results_path = report_dir / "probe-results.json"
    if not probe_results_path.exists():
        with probe_results_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": generated_at,
                    "status": "pending",
                    "notes": "Populate after executing loader probes (NCNN/OpenCV/MNN/TensorRT as applicable).",
                    "results": [],
                },
                f,
                indent=2,
            )

    print(f"Generated: {inventory_path}")
    print(f"Generated: {compatibility_path}")
    print(f"Generated: {fingerprints_path}")
    print(f"Generated: {probe_results_path}")
    print(f"Generated binwalk reports in: {binwalk_report_dir}")
    print(f"Artifacts analyzed: {len(files)}")
    print(f"Artifacts excluded: {len(excluded_artifacts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
