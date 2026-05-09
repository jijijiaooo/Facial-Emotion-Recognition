"""
Load an older runtime-analysis result bundle and regenerate the runtime plots
with the current plotting code.
"""
import json
import os
import sys
import types
import numpy as np

# Create a lightweight fake `cv2` module in sys.modules to allow importing
# `revised_emotion_detection` in environments without OpenCV installed.
if 'cv2' not in sys.modules:
    fake_cv2 = types.SimpleNamespace()
    fake_cv2.data = types.SimpleNamespace(haarcascades='')
    sys.modules['cv2'] = fake_cv2

# Provide a minimal fake `tensorflow` with `keras` submodule to avoid heavy TF dependency at import time.
if 'tensorflow' not in sys.modules:
    fake_tf = types.SimpleNamespace()
    fake_tf.__version__ = '0.0'
    # minimal test helpers
    fake_test = types.SimpleNamespace()
    fake_test.is_built_with_cuda = lambda : False
    setattr(fake_tf, 'test', fake_test)
    # minimal config namespace
    fake_tf.config = types.SimpleNamespace(list_physical_devices=(lambda x: []))
    # fake keras with needed namespaces
    fake_keras = types.SimpleNamespace()
    fake_keras.saving = types.SimpleNamespace(register_keras_serializable=(lambda *a, **k: (lambda x: x)))
    fake_keras.utils = types.SimpleNamespace(register_keras_serializable=(lambda *a, **k: (lambda x: x)))
    # losses base class
    class _Loss(object):
        pass
    fake_keras.losses = types.SimpleNamespace(Loss=_Loss)
    fake_tf.keras = fake_keras
    sys.modules['tensorflow'] = fake_tf
    sys.modules['tf'] = fake_tf

from src.core.revised_emotion_detection import RevisedEmotionDetector

SOURCE_DIR = os.path.join(
    'evaluation_results',
    'revised_runtime_analysis_20260331_032956',
)
OUTPUT_DIR = os.path.join(
    'evaluation_results',
    'revised_runtime_analysis_old_data_test',
)

with open(os.path.join(SOURCE_DIR, 'runtime_analysis.json'), 'r') as f:
    payload = json.load(f)

processing_speed = payload['processing_speed']
det = object.__new__(RevisedEmotionDetector)
det.prediction_times_sec = payload['per_trial_processing_time_sec']
det.prediction_trial_count = int(processing_speed['trial_count'])
det.total_prediction_time_sec = float(processing_speed['total_prediction_time_sec'])
det.total_predictions = int(processing_speed['samples'])
det.emotion_counts = payload.get('emotion_counts', {})
det.frame_elapsed_time_sec = payload['runtime_time_series']['elapsed_time_sec']
det.frame_instant_fps = payload['runtime_time_series']['instant_fps']
det.frame_detected_faces = payload['runtime_time_series']['detected_faces']
det.session_start_time = 0.0
det.model_path = payload.get('model_path', 'models/dummy_model.keras')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Running runtime analysis from old data:', SOURCE_DIR)
det.save_runtime_analysis(output_dir=OUTPUT_DIR)
print('\nSaved runtime analysis to:', OUTPUT_DIR)
print('\nFiles:')
for fn in sorted(os.listdir(OUTPUT_DIR)):
    print(' -', fn)
