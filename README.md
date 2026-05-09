# Facial Emotion Recognition System

This repository contains a facial emotion recognition platform with companion-style applications, real-time camera analysis, and supporting utilities for evaluation, deployment, and dataset management.

## Overview

The project combines emotion detection, interactive companion experiences, and multiple deployment targets. It includes application code, API endpoints, model assets, deployment scripts, and evaluation outputs for experimentation and analysis.

## Key Capabilities

- Real-time facial emotion detection from camera input and streaming sources
- Interactive companion applications with conversational and activity-based interactions
- API support for emotion inference and RTSP or SRT stream handling
- Training and evaluation resources for revised dataset workflows
- Deployment helpers for Docker, Azure App Service, Azure Container Instances, and Azure Machine Learning

## Repository Layout

- `apps/` and `android/`: companion application clients and mobile integration
- `api/`: Python API entry points and stream-processing helpers
- `src/`: core application and model code
- `models/`: stored model artifacts
- `tools/`: utility scripts used across the project
- `data/`: dataset files and training inputs
- `evaluation_results/`: evaluation runs, audits, and analysis outputs
- `data_quality_report/`: dataset quality reports and diagnostics

## Requirements

- Python 3.9 or newer
- A virtual environment for local development
- Webcam or supported video stream for live emotion detection
- Optional GPU acceleration for training or heavy inference workloads

## Setup

1. Create and activate a virtual environment.
2. Install the required dependencies from `requirements.txt` or the relevant API and enhancement requirements files.
3. Run the application, API, or companion entry point that matches your workflow.

Example:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_latest_app.py
```

## Usage

- Use the companion applications for interactive emotion-aware experiences.
- Use the API for programmatic inference and stream processing.
- Use the evaluation folders to review experiment results and generated reports.
- Use the deployment scripts when packaging the project for Docker or Azure targets.

## Documentation

Additional project-specific guidance is available in the following files:

- `QUICKSTART.md`
- `QUICK_REFERENCE.md`
- `DEPLOYMENT_GUIDE.md`
- `AZURE_DEPLOYMENT_GUIDE.md`
- `TRAIN_REVISED_DATASET.md`

## License

See `LICENSE` for the project license.
