#!/usr/bin/env python3
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: replot_runtime_variability.py <runtime_analysis_dir>")
        return 1

    runtime_dir = sys.argv[1]
    json_path = os.path.join(runtime_dir, "runtime_analysis.json")
    out_path = os.path.join(runtime_dir, "processing_time_variability.png")

    with open(json_path, "r") as f:
        payload = json.load(f)

    trial_times_sec = payload.get("per_trial_processing_time_sec", [])
    if not trial_times_sec:
        print("No per_trial_processing_time_sec found in runtime_analysis.json")
        return 1

    times_ms = np.array(trial_times_sec, dtype=np.float64) * 1000.0
    trial_idx = np.arange(1, len(times_ms) + 1)

    mean_ms = float(np.mean(times_ms))
    median_ms = float(np.median(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))
    p99_ms = float(np.percentile(times_ms, 99))
    min_ms = float(np.min(times_ms))
    max_ms = float(np.max(times_ms))

    focus_upper_ms = max(p99_ms * 1.15, mean_ms * 1.4)
    outlier_mask = times_ms > focus_upper_ms
    outlier_count = int(np.sum(outlier_mask))
    display_times_ms = np.minimum(times_ms, focus_upper_ms)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    warmup_trials = min(20, len(times_ms))
    if warmup_trials > 0:
        axes[0].axvspan(
            1,
            warmup_trials,
            color="#f1c40f",
            alpha=0.12,
            label=f"Warm-up (first {warmup_trials} trials)",
        )

    axes[0].plot(
        trial_idx,
        display_times_ms,
        color="#2980b9",
        linewidth=1.1,
        alpha=0.9,
        label="Per-trial time",
    )
    axes[0].axhline(mean_ms, color="#c0392b", linestyle="--", linewidth=1.6, label=f"Mean: {mean_ms:.2f} ms")
    axes[0].axhline(median_ms, color="#27ae60", linestyle="--", linewidth=1.4, label=f"Median: {median_ms:.2f} ms")
    axes[0].axhline(p95_ms, color="#8e44ad", linestyle="--", linewidth=1.4, label=f"P95: {p95_ms:.2f} ms")

    if outlier_count > 0:
        axes[0].scatter(
            trial_idx[outlier_mask],
            np.full(outlier_count, focus_upper_ms),
            color="#e74c3c",
            marker="x",
            s=35,
            label=f"Outliers above axis: {outlier_count}",
        )

    axes[0].set_title("Per-Trial Processing Time (Outlier-Aware View)")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Processing time (ms)")
    axes[0].set_ylim(0, focus_upper_ms * 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)

    focused_times_ms = times_ms[times_ms <= focus_upper_ms]
    p01_ms = float(np.percentile(focused_times_ms, 1)) if len(focused_times_ms) > 0 else min_ms
    p99_focus_ms = float(np.percentile(focused_times_ms, 99)) if len(focused_times_ms) > 0 else max_ms
    half_range = max(median_ms - p01_ms, p99_focus_ms - median_ms, 0.25)
    x_left = median_ms - half_range * 1.1
    x_right = median_ms + half_range * 1.1
    axes[1].hist(
        focused_times_ms,
        bins=min(32, max(10, len(focused_times_ms) // 8)),
        color="#16a085",
        alpha=0.82,
        edgecolor="white",
    )
    axes[1].axvline(mean_ms, color="#c0392b", linestyle="--", linewidth=1.6, label=f"Mean: {mean_ms:.2f} ms")
    axes[1].axvline(median_ms, color="#27ae60", linestyle="--", linewidth=1.4, label=f"Median: {median_ms:.2f} ms")
    axes[1].axvline(p95_ms, color="#8e44ad", linestyle="--", linewidth=1.4, label=f"P95: {p95_ms:.2f} ms")
    axes[1].set_title("Processing Time Distribution (Focused Range)")
    axes[1].set_xlabel("Processing time (ms)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(x_left, x_right)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)

    stats_text = (
        f"N={len(times_ms)}\n"
        f"Min={min_ms:.2f} ms\n"
        f"Max={max_ms:.2f} ms\n"
        f"P99={p99_ms:.2f} ms\n"
        f"Outliers>{focus_upper_ms:.2f} ms: {outlier_count}"
    )
    axes[1].text(
        0.02,
        0.98,
        stats_text,
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#bdc3c7"),
    )

    model_name = os.path.basename(payload.get("model_path", "model"))
    plt.suptitle(f"Processing-Time Variability - {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
