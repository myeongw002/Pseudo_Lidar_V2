"""Timing helpers for config-driven pipeline runners."""

import csv
import json
import time
from datetime import datetime
from pathlib import Path


TIMING_CSV_COLUMNS = [
    "stage",
    "action",
    "started_at",
    "ended_at",
    "duration_sec",
    "time_per_frame_sec",
    "processed_frame_count",
    "process_count",
    "subsampling_enabled",
    "average_correction_nodes",
    "correction_time_total_sec",
    "correction_time_per_frame_sec",
    "status_total",
    "status_existing",
    "status_missing",
    "command",
    "error",
]


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def begin_stage():
    return {
        "started_at": now_iso(),
        "_perf_start": time.perf_counter(),
    }


def finish_stage(result, timer, error=None):
    ended_at = now_iso()
    duration_sec = time.perf_counter() - timer["_perf_start"]
    result["started_at"] = timer["started_at"]
    result["ended_at"] = ended_at
    result["duration_sec"] = round(duration_sec, 6)
    if error is not None:
        result["error"] = str(error)
    return result


def instant_stage(stage, action, **extra):
    timer = begin_stage()
    result = {"stage": stage, "action": action, **extra}
    return finish_stage(result, timer)


def stage_status_values(result):
    status = result.get("status_before") or {}
    return {
        "status_total": status.get("total", ""),
        "status_existing": status.get("existing", ""),
        "status_missing": status.get("missing", ""),
    }


def timing_csv_row(result):
    row = {
        "stage": result.get("stage", ""),
        "action": result.get("action", ""),
        "started_at": result.get("started_at", ""),
        "ended_at": result.get("ended_at", ""),
        "duration_sec": result.get("duration_sec", ""),
        "time_per_frame_sec": result.get("time_per_frame_sec", ""),
        "processed_frame_count": result.get("processed_frame_count", ""),
        "process_count": result.get("process_count", ""),
        "subsampling_enabled": result.get("subsampling_enabled", ""),
        "average_correction_nodes": result.get("average_correction_nodes", ""),
        "correction_time_total_sec": result.get("correction_time_total_sec", ""),
        "correction_time_per_frame_sec": result.get("correction_time_per_frame_sec", ""),
        "command": " ".join(str(part) for part in result.get("command") or []),
        "error": result.get("error", ""),
    }
    row.update(stage_status_values(result))
    return row


def aggregate_timing(stage_results):
    measured = [row for row in stage_results if "duration_sec" in row]
    by_action = {}
    for row in measured:
        action = row.get("action", "unknown")
        item = by_action.setdefault(
            action,
            {"count": 0, "total_duration_sec": 0.0, "max_duration_sec": 0.0},
        )
        duration = float(row.get("duration_sec") or 0.0)
        item["count"] += 1
        item["total_duration_sec"] += duration
        item["max_duration_sec"] = max(item["max_duration_sec"], duration)

    for item in by_action.values():
        count = item["count"]
        item["total_duration_sec"] = round(item["total_duration_sec"], 6)
        item["max_duration_sec"] = round(item["max_duration_sec"], 6)
        item["mean_duration_sec"] = round(item["total_duration_sec"] / count, 6) if count else 0.0

    slowest = sorted(
        (
            {
                "stage": row.get("stage"),
                "action": row.get("action"),
                "duration_sec": float(row.get("duration_sec") or 0.0),
            }
            for row in measured
        ),
        key=lambda item: item["duration_sec"],
        reverse=True,
    )
    return {
        "stage_count": len(measured),
        "total_stage_duration_sec": round(
            sum(float(row.get("duration_sec") or 0.0) for row in measured), 6
        ),
        "by_action": by_action,
        "slowest_stages": slowest[:10],
    }


def write_timing_files(stage_results, csv_path, summary_path, metadata=None):
    csv_path = Path(csv_path)
    summary_path = Path(summary_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [timing_csv_row(result) for result in stage_results if "duration_sec" in result]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "metadata": metadata or {},
        "summary": aggregate_timing(stage_results),
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return csv_path, summary_path, payload


def print_timing_summary(stage_results, csv_path=None, summary_path=None):
    summary = aggregate_timing(stage_results)
    print("\n[Stage Timing]")
    print(f"stage_count: {summary['stage_count']}")
    print(f"total_stage_duration_sec: {summary['total_stage_duration_sec']}")
    for action, values in sorted(summary["by_action"].items()):
        print(
            f"{action}: count={values['count']}, "
            f"total={values['total_duration_sec']}, "
            f"mean={values['mean_duration_sec']}"
        )
    if summary["slowest_stages"]:
        top = summary["slowest_stages"][0]
        print(f"slowest: {top['stage']} ({top['action']}), {top['duration_sec']} sec")
    if csv_path is not None:
        print(f"timing_csv: {csv_path}")
    if summary_path is not None:
        print(f"timing_summary: {summary_path}")
