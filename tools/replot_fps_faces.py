#!/usr/bin/env python3
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: replot_fps_faces.py <runtime_analysis_dir>")
        return 1

    runtime_dir = sys.argv[1]
    json_path = os.path.join(runtime_dir, "runtime_analysis.json")
    out_path = os.path.join(runtime_dir, "fps_vs_detected_faces.png")

    with open(json_path, "r") as f:
        payload = json.load(f)

    ts = payload.get("runtime_time_series", {})
    elapsed = np.array(ts.get("elapsed_time_sec", []), dtype=np.float64)
    fps = np.array(ts.get("instant_fps", []), dtype=np.float64)
    faces = np.array(ts.get("detected_faces", []), dtype=np.float64)

    if len(elapsed) == 0 or len(fps) == 0 or len(faces) == 0:
        print("runtime_time_series is missing in runtime_analysis.json. Run revised_emotion_detection.py again to capture frame-level time-series.")
        return 1

    window = min(15, len(fps))
    if window > 1:
        smooth = np.convolve(fps, np.ones(window) / window, mode="same")
    else:
        smooth = fps

    fig, ax1 = plt.subplots(figsize=(10.5, 6))
    ax2 = ax1.twinx()

    ax1.plot(elapsed, fps, color="#4c8eda", alpha=0.35, linewidth=1.0, label="Instant FPS")
    ax1.plot(elapsed, smooth, color="#2f6fb6", linewidth=2.2, label="Smoothed FPS")
    ax1.set_xlabel("Time (Seconds)", fontweight="bold")
    ax1.set_ylabel("Frame Rate Per Second", color="#2f6fb6", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#2f6fb6")
    ax1.grid(alpha=0.25)

    ax2.plot(elapsed, faces, color="#d14b46", linewidth=2.0, label="Detected Faces", drawstyle="steps-post")
    ax2.set_ylabel("Number of Detected Faces", color="#d14b46", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#d14b46")

    mean_fps = float(np.mean(fps)) if len(fps) > 0 else 0.0
    mean_faces = float(np.mean(faces)) if len(faces) > 0 else 0.0
    ax1.text(0.02, 0.95, f"Mean FPS: {mean_fps:.2f}", transform=ax1.transAxes, color="#2f6fb6", fontsize=10, fontweight="bold")
    ax2.text(0.62, 0.95, f"Mean Faces: {mean_faces:.2f}", transform=ax2.transAxes, color="#d14b46", fontsize=10, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", framealpha=0.9)

    plt.title("FPS and Detected Faces vs Time", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
