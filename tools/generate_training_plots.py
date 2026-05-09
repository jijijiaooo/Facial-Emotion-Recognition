#!/usr/bin/env python3
import json
import os
import runpy

ROOT = os.path.dirname(os.path.dirname(__file__))
EVAL_SCRIPT = os.path.join(ROOT, 'evaluation_results', 'revised_cnn_eval_20260223_121202_tta10', 'evaluate_revised_cnn.py')
HISTORY_PATH = os.path.join(ROOT, 'logs', 'training_revised_20260220_030450', 'training_history.json')
OUTPUT_DIR = os.path.join(ROOT, 'evaluation_results', 'revised_cnn_eval_20260223_121202_tta10')
MODEL_TITLE = 'emotion_revised_cnn_20260220_030450.keras (TTAx10)'

print('Loading evaluation module...')
mod_globals = runpy.run_path(EVAL_SCRIPT)
_plot_training_curves = mod_globals.get('_plot_training_curves')
if _plot_training_curves is None:
    raise RuntimeError('Could not find _plot_training_curves in the evaluation script')

print(f'Loading history: {HISTORY_PATH}')
with open(HISTORY_PATH, 'r') as f:
    payload = json.load(f)
history = payload.get('history', payload)

print('Generating plots...')
result = _plot_training_curves(history, OUTPUT_DIR, MODEL_TITLE)
print('Done. Result:')
print(result)
