# Quick Evaluation Guide

## TL;DR - Run Evaluation Fast

### Quick Mode (Recommended - ~2-5 minutes)
```bash
python3 evaluate_system.py --quick
```
- Evaluates **50 samples per emotion** (~350 total)
- Fast and accurate results
- Best for rapid testing and iteration

### Fast Mode (~5-10 minutes)
```bash
python3 evaluate_system.py --fast
```
- Evaluates **100 samples per emotion** (~700 total)
- More comprehensive results
- Good balance of speed and accuracy

### Balanced Mode (Default - ~10-20 minutes)
```bash
python3 evaluate_system.py
```
- Evaluates **200 samples per emotion** (~1400 total)
- High accuracy without processing all 7000+ images

### Full Evaluation (~30-60 minutes)
```bash
python3 evaluate_system.py --max-per-class 10000
```
- Processes **all available test images** (~7000+)
- Most comprehensive but slow
- Only needed for final validation

## Advanced Options

### Custom Sampling
```bash
# Evaluate exactly 500 images total
python3 evaluate_system.py --sample-size 500

# Evaluate 75-150 samples per class
python3 evaluate_system.py --min-per-class 75 --max-per-class 150

# Custom test data path
python3 evaluate_system.py --test-path path/to/test/data
```

### All Options
```bash
python3 evaluate_system.py --help
```

## Why Sampling is Better

### Statistical Validity
- **50 samples per class** = 95% confidence level with ±14% margin of error
- **100 samples per class** = 95% confidence level with ±10% margin of error
- **200 samples per class** = 95% confidence level with ±7% margin of error

### Speed vs Accuracy Trade-off

| Mode | Samples | Time | Accuracy | Use Case |
|------|---------|------|----------|----------|
| Quick | 350 | 2-5 min | ±14% | Development, quick tests |
| Fast | 700 | 5-10 min | ±10% | Regular validation |
| Balanced | 1400 | 10-20 min | ±7% | Pre-deployment checks |
| Full | 7000+ | 30-60 min | ±4% | Final validation only |

### Why Not Always Use Full?
1. **Diminishing returns**: After 200 samples per class, accuracy improvement is minimal
2. **Time waste**: 10-20 minutes vs 60 minutes for marginal gain
3. **Iteration speed**: Faster evaluation = faster debugging cycle
4. **Resource usage**: Lower memory and CPU usage

## Understanding the Results

After evaluation completes, check the results folder for:

### Key Files
1. **performance_dashboard.png** - Quick visual overview
2. **evaluation_summary.json** - Metrics in JSON format
3. **confusion_matrix_normalized.png** - See which emotions get confused

### Quick Interpretation
- **Accuracy > 70%**: Good system
- **Accuracy > 80%**: Great system
- **Accuracy > 90%**: Excellent system

Check the confusion matrix to see specific emotion pairs that get mixed up.

## Optimization Tips

### For Fastest Evaluation
1. Use `--quick` mode
2. Ensure models are cached (run once first)
3. Close other applications to free CPU

### For Most Accurate Results
1. Use balanced mode (default)
2. Run multiple times and average results
3. Check for class imbalance in dataset distribution

### When to Use Each Mode

**Quick Mode** - Use when:
- Debugging code changes
- Testing new features
- Rapid iteration during development
- Just want to verify system works

**Fast Mode** - Use when:
- Validating bug fixes
- Testing emotion pattern changes
- Weekly performance checks
- Comparing different configurations

**Balanced Mode** - Use when:
- Pre-deployment testing
- Monthly performance reports
- Documenting system capabilities
- Comparing with other systems

**Full Mode** - Use when:
- Final validation before production
- Academic paper/publication
- Official benchmarking
- Regulatory requirements

## Example Workflow

### Daily Development
```bash
# Make code changes
vim src/core/simple_emotion_detection.py

# Quick test
python3 evaluate_system.py --quick

# If looks good, run fast validation
python3 evaluate_system.py --fast
```

### Pre-Deployment
```bash
# Full balanced evaluation
python3 evaluate_system.py

# Review results in evaluation_results/latest/
# Check performance_dashboard.png
# Verify accuracy meets requirements
```

### Monthly Report
```bash
# Run balanced evaluation
python3 evaluate_system.py

# Save results to dated folder
cp -r evaluation_results/evaluation_* reports/monthly_$(date +%Y%m)/
```

## Interpreting Evaluation Time

Expected times on typical hardware:

- **Quick (350 images)**: 2-5 minutes
  - Loading: 30s
  - Processing: 1-3 min
  - Visualization: 30s

- **Fast (700 images)**: 5-10 minutes
  - Loading: 1 min
  - Processing: 3-7 min
  - Visualization: 1 min

- **Balanced (1400 images)**: 10-20 minutes
  - Loading: 2 min
  - Processing: 6-15 min
  - Visualization: 2 min

If evaluation takes significantly longer, check:
1. Model loading issues
2. Disk I/O bottlenecks
3. CPU usage by other applications
4. Memory swapping

## Pro Tips

1. **Start with quick mode** - Get immediate feedback
2. **Use fast mode for validation** - Good balance
3. **Reserve full mode for final testing** - Not needed often
4. **Compare evaluations** - Track performance over time
5. **Focus on confusion matrix** - More informative than raw accuracy
6. **Check per-class metrics** - Some emotions may be harder than others

## Sample Output

```
🚀 QUICK MODE: Evaluating 50 samples per class (~350 total)

Loading test images...
  Found 1290 images for angry
  Sampling 50/1290 images for angry
  Found 160 images for disgust
  Using all 160 images for disgust
  ...

Total images loaded for evaluation: 350
This is 4.9% of available data

Running predictions...
  Batch 1/7: Processing images 0-49...
  Batch 2/7: Processing images 50-99...
  ...

✓ Predictions complete! Failed: 0/350

Overall Accuracy: 0.7486 (74.86%)

Results saved to: evaluation_results/evaluation_20251005_143022
Overall Accuracy: 0.7486 (74.86%)
```
