# Emotion Detection System Evaluation

This directory contains the evaluation scripts and results for the Facial Emotion Recognition system.

## Overview

The evaluation system tests the emotion detection model on the RAF-DB test dataset and generates comprehensive performance metrics including:

- **Precision**: How many of the predicted emotions were correct
- **Recall**: How many of the actual emotions were detected
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions vs actual labels
- **Overall Accuracy**: Percentage of correct predictions

## Running the Evaluation

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install opencv-python numpy matplotlib seaborn scikit-learn tensorflow
```

### Run Evaluation

From the project root directory, run:

```bash
python evaluate_system.py
```

This will:
1. Load all test images from `data/raf_db/processed/test/`
2. Run emotion detection on each image
3. Calculate performance metrics
4. Generate visualizations
5. Save all results to `evaluation_results/evaluation_YYYYMMDD_HHMMSS/`

## Output Files

Each evaluation run creates a timestamped folder containing:

### Metric Files
- **`evaluation_summary.json`**: Complete metrics in JSON format
  - Overall accuracy
  - Per-emotion precision, recall, F1-score
  - Macro and weighted averages
  
- **`classification_report.txt`**: Detailed text report with all metrics

- **`confusion_matrix.csv`**: Confusion matrix in CSV format for Excel/analysis

- **`raw_predictions.json`**: Raw prediction data for custom analysis

### Visualizations
- **`confusion_matrix.png`**: Heatmap showing prediction counts
- **`confusion_matrix_normalized.png`**: Percentage-based confusion matrix
- **`metrics_comparison.png`**: Bar chart comparing precision, recall, and F1-score
- **`dataset_distribution.png`**: Distribution of test samples per emotion
- **`performance_dashboard.png`**: Complete overview dashboard

## Understanding the Metrics

### Precision
**Definition**: Of all the times we predicted emotion X, how often were we correct?

**Formula**: True Positives / (True Positives + False Positives)

**Example**: If we predicted "Happy" 100 times and 85 were actually happy, precision = 0.85

### Recall (Sensitivity)
**Definition**: Of all the actual emotion X instances, how many did we detect?

**Formula**: True Positives / (True Positives + False Negatives)

**Example**: If there were 100 happy faces and we detected 80, recall = 0.80

### F1-Score
**Definition**: Balanced measure combining precision and recall

**Formula**: 2 × (Precision × Recall) / (Precision + Recall)

**Use**: Good overall metric when you want to balance precision and recall

### Confusion Matrix
Shows the relationship between predicted and actual emotions:
- **Diagonal values**: Correct predictions
- **Off-diagonal values**: Misclassifications
- **Row totals**: Actual emotion counts
- **Column totals**: Predicted emotion counts

## Interpreting Results

### Good Performance Indicators
- Overall accuracy > 0.70 (70%)
- F1-scores > 0.60 for most emotions
- High values on confusion matrix diagonal
- Low confusion between dissimilar emotions

### Common Issues to Watch For
- **Class imbalance**: Some emotions harder to detect due to fewer samples
- **Confusion patterns**: Check which emotions get confused (e.g., Fear vs Surprise)
- **Low recall**: Missing actual instances of emotion
- **Low precision**: False positives for emotion

## Example Results Structure

```
evaluation_results/
└── evaluation_20250105_143022/
    ├── evaluation_summary.json
    ├── classification_report.txt
    ├── confusion_matrix.csv
    ├── raw_predictions.json
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── metrics_comparison.png
    ├── dataset_distribution.png
    └── performance_dashboard.png
```

## Customization

To modify the evaluation:

1. **Change test data path**: Edit `test_data_path` in `evaluate_system.py`
2. **Add custom metrics**: Extend the `EmotionSystemEvaluator` class
3. **Modify visualizations**: Update the `generate_visualizations()` method
4. **Change output format**: Modify the `save_results()` method

## Troubleshooting

### "No test images found"
- Check that `data/raf_db/processed/test/` exists
- Verify subdirectories for each emotion contain images

### Import errors
- Install missing packages: `pip install -r requirements.txt`
- Check Python version (requires 3.7+)

### Memory errors
- Process images in smaller batches
- Reduce visualization DPI in `generate_visualizations()`

### Low performance
- Check model files are loaded correctly
- Verify test images are properly formatted
- Review recent changes to emotion detection logic

## Notes

- Evaluation uses the same detection pipeline as the live system
- Results may vary based on:
  - Model versions loaded
  - Detection thresholds
  - Image quality and preprocessing
  - Recent code changes to emotion patterns

## Version History

- **2025-10-05**: Initial evaluation system created
  - Updated Fear/Surprise detection patterns
  - Fear now focuses on eye widening
  - Surprise requires mouth opening + raised brows
