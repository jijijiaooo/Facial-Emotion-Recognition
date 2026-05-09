#!/usr/bin/env python3
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: replot_runtime_fps.py <runtime_analysis_dir>")
        return 1

    runtime_dir = sys.argv[1]
    json_path = os.path.join(runtime_dir, "runtime_analysis.json")
    out_path = os.path.join(runtime_dir, "fps_analysis.png")

    with open(json_path, "r") as f:
        payload = json.load(f)

    trial_times_sec = payload.get("per_trial_processing_time_sec", [])
    speed = payload.get("processing_speed", {})
    if not trial_times_sec:
        print("No per_trial_processing_time_sec found in runtime_analysis.json")
        return 1

    per_trial_fps = 1.0 / np.array(trial_times_sec, dtype=np.float64)
    trial_idx = np.arange(1, len(per_trial_fps) + 1)
    window = min(20, len(per_trial_fps))

    if window > 1:
        rolling_fps = np.convolve(per_trial_fps, np.ones(window) / window, mode="valid")
        rolling_x = np.arange(window, len(per_trial_fps) + 1)
    else:
        rolling_fps = per_trial_fps
        rolling_x = trial_idx

    mean_fps = float(speed.get("mean_fps", np.mean(per_trial_fps)))
    median_fps = float(speed.get("median_fps", np.median(per_trial_fps)))
    std_fps = float(speed.get("std_fps", np.std(per_trial_fps)))
    p05_fps = float(speed.get("p05_fps", np.percentile(per_trial_fps, 5)))
    p95_fps = float(speed.get("p95_fps", np.percentile(per_trial_fps, 95)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    axes[0].axhspan(10, 30, color="#9be7a8", alpha=0.20, label="Real-time range (10-30 FPS)")
    axes[0].plot(trial_idx, per_trial_fps, color="#5dade2", linewidth=0.9, alpha=0.55, label="Per-trial FPS")
    axes[0].plot(rolling_x, rolling_fps, color="#1f618d", linewidth=2.0, label=f"Rolling FPS (window={window})")
    axes[0].axhline(mean_fps, color="#c0392b", linestyle="--", linewidth=1.6, label=f"Mean FPS: {mean_fps:.2f}")
    axes[0].axhline(median_fps, color="#27ae60", linestyle="--", linewidth=1.4, label=f"Median FPS: {median_fps:.2f}")
    axes[0].set_title("FPS Stability Over Trials")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Frames per second")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].hist(per_trial_fps, bins=min(32, max(10, len(per_trial_fps) // 8)), color="#48c9b0", alpha=0.82, edgecolor="white")
    axes[1].axvline(mean_fps, color="#c0392b", linestyle="--", linewidth=1.6, label=f"Mean: {mean_fps:.2f}")
    axes[1].axvline(median_fps, color="#27ae60", linestyle="--", linewidth=1.4, label=f"Median: {median_fps:.2f}")
    axes[1].axvline(p05_fps, color="#7d3c98", linestyle="--", linewidth=1.2, label=f"P05: {p05_fps:.2f}")
    axes[1].axvline(p95_fps, color="#8e44ad", linestyle="--", linewidth=1.2, label=f"P95: {p95_fps:.2f}")
    axes[1].set_title("FPS Distribution")
    axes[1].set_xlabel("Frames per second")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)

    stat_text = (
        f"N={len(per_trial_fps)}\n"
        f"Std={std_fps:.2f}\n"
        f"P05={p05_fps:.2f}\n"
        f"P95={p95_fps:.2f}\n"
        "Real-time target: >=10 FPS"
    )
    axes[1].text(
        0.02,
        0.98,
        stat_text,
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#bdc3c7"),
    )

    model_name = os.path.basename(payload.get("model_path", "model"))
    plt.suptitle(f"FPS Analysis - {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
