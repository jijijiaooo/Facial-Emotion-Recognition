#!/usr/bin/env python3
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: replot_cumulative_frames.py <runtime_analysis_dir>")
        return 1

    runtime_dir = sys.argv[1]
    json_path = os.path.join(runtime_dir, "runtime_analysis.json")
    out_path = os.path.join(runtime_dir, "cumulative_frames_over_time.png")

    with open(json_path, "r") as f:
        payload = json.load(f)

    trial_times_sec = payload.get("per_trial_processing_time_sec", [])
    if not trial_times_sec:
        print("No per_trial_processing_time_sec found in runtime_analysis.json")
        return 1

    times = np.array(trial_times_sec, dtype=np.float64)
    elapsed_sec = np.cumsum(times)
    cumulative_frames = np.arange(1, len(times) + 1)

    avg_fps = len(times) / float(np.sum(times)) if np.sum(times) > 0 else 0.0
    ref_frames = avg_fps * elapsed_sec

    plt.figure(figsize=(10, 6))
    plt.plot(elapsed_sec, cumulative_frames, color="#1f77b4", linewidth=2.2, label="Observed cumulative frames")
    plt.plot(elapsed_sec, ref_frames, color="#d62728", linestyle="--", linewidth=1.6, label=f"Average-rate reference ({avg_fps:.2f} FPS)")

    plt.title("Cumulative Frames vs Elapsed Time")
    plt.xlabel("Elapsed time (seconds)")
    plt.ylabel("Cumulative frames processed")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left")

    summary_text = (
        f"Total frames: {len(times)}\n"
        f"Total time: {elapsed_sec[-1]:.2f} s\n"
        f"Average FPS: {avg_fps:.2f}"
    )
    plt.text(
        0.02,
        0.98,
        summary_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#bdc3c7"),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
