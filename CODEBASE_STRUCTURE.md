# Codebase Structure

This repository is organized around a small set of focused areas so the codebase stays maintainable without moving or deleting files.

## Main Areas

- `src/`: Core emotion detection logic, model loading, and training helpers.
- `api/`: FastAPI entry points and stream-processing support.
- `apps/companions/`: Companion application interfaces for desktop and device-based clients.
- `android/`: Android integration assets.
- `models/`: Stored model artifacts.
- `tools/`: Plotting, reporting, and maintenance utilities.
- `data/`: Training and dataset inputs.
- `evaluation_results/`: Saved evaluation runs and analysis output.
- `data_quality_report/`: Generated quality reports and dataset diagnostics.

## Package Boundaries

- Treat `src/` as the primary home for reusable application logic.
- Keep API-specific code inside `api/`.
- Keep UI and companion code inside `apps/`.
- Keep scripts that generate reports or plots inside `tools/`.

## Organization Notes

- No files are removed by this pass.
- Existing scripts remain in place.
- The goal is to make imports, navigation, and future refactoring clearer.
