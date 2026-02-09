# Training on Revised Dataset (6 Emotion Classes)

## Overview
The revised dataset contains **6 emotion classes** (no 'surprise'):
- angry
- disgust
- fear
- happy
- neutral
- sad

## Quick Start

### Train Locally
```bash
# Basic training (default settings)
python src/core/train_revised_dataset.py

# Custom configuration
python src/core/train_revised_dataset.py \
  --img_size 128 \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.6
```

### Train on Azure ML

1. **Upload dataset to Azure ML Studio**:
   - Data → Create → Upload → Select `data/revised_dataset/`

2. **Create compute cluster**:
   - Compute → Compute clusters → New
   - Name: `gpu-cluster`
   - Size: `Standard_NC6` (GPU) or `Standard_D4s_v3` (CPU)

3. **Submit training job**:
   - Jobs → Create → Command job
   - Code: Upload `src/core/train_revised_dataset.py`
   - Command: `python train_revised_dataset.py --epochs 100`
   - Environment: `AzureML-tensorflow-2.16-cuda12`
   - Compute: `gpu-cluster`

## Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_dir` | `data/revised_dataset/train` | Training data directory |
| `--val_dir` | `data/revised_dataset/validation` | Validation data directory |
| `--img_size` | `112` | Input image size (112x112) |
| `--batch_size` | `64` | Batch size for training |
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `0.001` | Initial learning rate |
| `--dropout` | `0.5` | Dropout rate for dense layers |
| `--model_dir` | `models` | Directory to save trained models |
| `--logs_dir` | `logs/training_revised_*` | TensorBoard logs directory |
| `--use_mixed_precision` | `False` | Use mixed precision (faster on GPU) |
| `--warmup_epochs` | `5` | Learning rate warmup epochs |

## Training Examples

### Fast Training (Testing)
```bash
python src/core/train_revised_dataset.py \
  --epochs 20 \
  --batch_size 32 \
  --img_size 96
```

### High Accuracy (Production)
```bash
python src/core/train_revised_dataset.py \
  --epochs 150 \
  --batch_size 64 \
  --img_size 128 \
  --dropout 0.6 \
  --lr 0.0008 \
  --use_mixed_precision
```

### GPU Training (Faster)
```bash
python src/core/train_revised_dataset.py \
  --epochs 100 \
  --batch_size 128 \
  --use_mixed_precision
```

## Model Output

After training completes, you'll get:

1. **Trained model**: `models/emotion_revised_cnn_YYYYMMDD_HHMMSS.h5`
2. **Training logs**: `logs/training_revised_YYYYMMDD_HHMMSS/`
3. **Training history**: `logs/training_revised_YYYYMMDD_HHMMSS/training_history.json`

## Monitoring Training

### View TensorBoard
```bash
tensorboard --logdir=logs/training_revised_YYYYMMDD_HHMMSS
```

Then open: http://localhost:6006

### Check Training History
```python
import json

with open('logs/training_revised_YYYYMMDD_HHMMSS/training_history.json') as f:
    history = json.load(f)

print(f"Best validation accuracy: {max(history['history']['val_accuracy']):.4f}")
print(f"Final training accuracy: {history['history']['accuracy'][-1]:.4f}")
```

## Expected Results

| Dataset Size | GPU Training Time | Expected Accuracy |
|-------------|-------------------|-------------------|
| Small (~5K) | 15-30 min | 65-75% |
| Medium (~15K) | 30-60 min | 75-85% |
| Large (~30K+) | 1-2 hours | 85-92% |

## Differences from Combined Dataset Training

| Feature | Combined Dataset (7 classes) | Revised Dataset (6 classes) |
|---------|------------------------------|----------------------------|
| Classes | angry, disgust, fear, happy, neutral, sad, **surprise** | angry, disgust, fear, happy, neutral, sad |
| Script | `train_combined_dataset.py` | `train_revised_dataset.py` |
| Default data path | `data/combined_dataset_complete/` | `data/revised_dataset/` |
| Output classes | 7 | 6 |
| Model name | `emotion_combined_cnn_*.h5` | `emotion_revised_cnn_*.h5` |

## Deployment

After training, deploy the model:

### Option 1: Update Azure App Service
```bash
# Copy trained model
cp models/emotion_revised_cnn_YYYYMMDD_HHMMSS.h5 models/emotion_revised_6classes.h5

# Update API to use new model
# Edit api/main.py:
# MODEL_PATH = 'models/emotion_revised_6classes.h5'
# EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# Rebuild and push Docker image
docker build --platform linux/amd64 -f Dockerfile.api -t emotion-detection-api .
docker tag emotion-detection-api emotiondetectionpelio-gyb9hbcygcchh6b3.azurecr.io/emotion-detection-api:latest
docker push emotiondetectionpelio-gyb9hbcygcchh6b3.azurecr.io/emotion-detection-api:latest

# Restart App Service
az webapp restart --name emotion-detection-api-g9budncvekdgewbk --resource-group emotion-detection-pelio
```

### Option 2: Test Locally First
```bash
# Create test script
python -c "
from tensorflow import keras
import numpy as np

model = keras.models.load_model('models/emotion_revised_cnn_YYYYMMDD_HHMMSS.h5')
print('Model loaded successfully!')
print(f'Input shape: {model.input_shape}')
print(f'Output shape: {model.output_shape}')
print(f'Classes: 6 emotions')
"
```

## Troubleshooting

**Issue**: Out of memory error
```bash
# Solution: Reduce batch size
python src/core/train_revised_dataset.py --batch_size 32
```

**Issue**: Training too slow
```bash
# Solution: Enable mixed precision
python src/core/train_revised_dataset.py --use_mixed_precision
```

**Issue**: Overfitting (train acc >> val acc)
```bash
# Solution: Increase dropout
python src/core/train_revised_dataset.py --dropout 0.7
```

**Issue**: Underfitting (low accuracy on both)
```bash
# Solution: Increase epochs and reduce dropout
python src/core/train_revised_dataset.py --epochs 150 --dropout 0.4
```
