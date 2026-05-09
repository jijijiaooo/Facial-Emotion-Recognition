#!/usr/bin/env python3
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: replot_recommended_fps.py <runtime_analysis_dir>")
        return 1

    runtime_dir = sys.argv[1]
    json_path = os.path.join(runtime_dir, "runtime_analysis.json")
    out_path = os.path.join(runtime_dir, "fps_recommended_analysis.png")

    with open(json_path, "r") as f:
        payload = json.load(f)

    ts = payload.get("runtime_time_series", {})
    elapsed = np.array(ts.get("elapsed_time_sec", []), dtype=np.float64)
    fps = np.array(ts.get("instant_fps", []), dtype=np.float64)

    if len(elapsed) == 0 or len(fps) == 0:
        # Fallback to per-trial processing times
        trial_times = np.array(payload.get("per_trial_processing_time_sec", []), dtype=np.float64)
        if len(trial_times) == 0:
            print("No runtime time-series or per-trial timing data found")
            return 1
        elapsed = np.cumsum(trial_times)
        fps = 1.0 / trial_times

    # Remove non-positive values to avoid percentile artifacts
    valid = fps > 0
    elapsed = elapsed[valid]
    fps = fps[valid]
    if len(fps) == 0:
        print("No valid FPS values found")
        return 1

    mean_fps = float(np.mean(fps))
    median_fps = float(np.median(fps))
    std_fps = float(np.std(fps, ddof=0))
    p05 = float(np.percentile(fps, 5))
    p95 = float(np.percentile(fps, 95))

    window = min(20, len(fps))
    if window > 1:
        rolling = np.convolve(fps, np.ones(window) / window, mode="same")
    else:
        rolling = fps

    # ECDF for robust percentile visualization
    x_sorted = np.sort(fps)
    y_ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1) FPS over time
    ax = axes[0, 0]
    ax.axhspan(10, 30, color="#9be7a8", alpha=0.18, label="Real-time target range (10-30 FPS)")
    ax.plot(elapsed, fps, color="#6fa8dc", linewidth=0.9, alpha=0.5, label="Instant FPS")
    ax.plot(elapsed, rolling, color="#1f4e79", linewidth=2.0, label=f"Rolling FPS (window={window})")
    ax.axhline(mean_fps, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Mean: {mean_fps:.2f}")
    ax.axhline(median_fps, color="#27ae60", linestyle="--", linewidth=1.3, label=f"Median: {median_fps:.2f}")
    ax.set_title("FPS Stability Over Time")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("FPS")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # 2) FPS histogram
    ax = axes[0, 1]
    ax.hist(fps, bins=min(32, max(10, len(fps) // 8)), color="#48c9b0", alpha=0.85, edgecolor="white")
    ax.axvline(mean_fps, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Mean: {mean_fps:.2f}")
    ax.axvline(median_fps, color="#27ae60", linestyle="--", linewidth=1.3, label=f"Median: {median_fps:.2f}")
    ax.axvline(p05, color="#7d3c98", linestyle="--", linewidth=1.2, label=f"P05: {p05:.2f}")
    ax.axvline(p95, color="#8e44ad", linestyle="--", linewidth=1.2, label=f"P95: {p95:.2f}")
    ax.set_title("FPS Distribution")
    ax.set_xlabel("FPS")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # 3) Cumulative frames vs time
    ax = axes[1, 0]
    frame_idx = np.arange(1, len(elapsed) + 1)
    ref_frames = mean_fps * elapsed
    ax.plot(elapsed, frame_idx, color="#1f77b4", linewidth=2.0, label="Observed cumulative frames")
    ax.plot(elapsed, ref_frames, color="#d62728", linestyle="--", linewidth=1.4, label=f"Average-rate reference ({mean_fps:.2f} FPS)")
    ax.set_title("Throughput Trend")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Cumulative frames")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # 4) ECDF percentile view
    ax = axes[1, 1]
    ax.plot(x_sorted, y_ecdf, color="#2e86c1", linewidth=2.0)
    ax.axvline(p05, color="#7d3c98", linestyle="--", linewidth=1.2, label=f"P05: {p05:.2f}")
    ax.axvline(p95, color="#8e44ad", linestyle="--", linewidth=1.2, label=f"P95: {p95:.2f}")
    ax.set_title("FPS ECDF (Percentile View)")
    ax.set_xlabel("FPS")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    summary = (
        f"N={len(fps)}  Mean={mean_fps:.2f}  Median={median_fps:.2f}  Std={std_fps:.2f}  "
        f"P05={p05:.2f}  P95={p95:.2f}"
    )
    fig.suptitle(f"Recommended FPS Analysis Dashboard\n{summary}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
