
# YOLO Snail Species Detection Research Data

**Generated:** 2025-06-24 15:23:34
**Research ID:** 20250624_152332

## Overview

This directory contains comprehensive research data for comparative analysis of YOLO architectures (v5, v8, v10, v11) applied to snail species detection and recognition.

### Dataset Information
- **Total Images:** 4,204
- **Classes:** 4 (bf, bsg, bss, unknown)
- **Training Images:** 3,213
- **Validation Images:** 918
- **Test Images:** 73
- **Training Epochs:** 300
- **Image Resolution:** 640x640 pixels

### Models Analyzed

#### YOLOv5
- **Accuracy:** 92.5%
- **mAP@0.5:** 78.9%
- **Model Size:** 160.0 MB
- **FPS:** 3.8
- **Architecture:** CSPDarknet53
- **Training Time:** 4.2h
- **Model Path:** `/content/drive/MyDrive/Colab Notebooks/v5.pt`

#### YOLOv8
- **Accuracy:** 95.4%
- **mAP@0.5:** 94.2%
- **Model Size:** 49.6 MB
- **FPS:** 5.0
- **Architecture:** CSPDarknet + C2f
- **Training Time:** 3.1h
- **Model Path:** `/content/drive/MyDrive/Colab Notebooks/v8.pt`

#### YOLOv10
- **Accuracy:** 98.7%
- **mAP@0.5:** 96.7%
- **Model Size:** 31.9 MB
- **FPS:** 4.5
- **Architecture:** RT-DETRs + Efficient backbone
- **Training Time:** 2.8h
- **Model Path:** `/content/drive/MyDrive/Colab Notebooks/v10.pt`

#### YOLOv11
- **Accuracy:** 97.8%
- **mAP@0.5:** 96.5%
- **Model Size:** 38.7 MB
- **FPS:** 4.7
- **Architecture:** C3k2 + SPPF
- **Training Time:** 2.9h
- **Model Path:** `/content/drive/MyDrive/Colab Notebooks/v11.pt`


## Data Files

### Training Data
- `combined_training_data_20250624_152332.csv`: All models training progression
- `yolo_research_data_20250624_152332.xlsx`: Excel file with multiple sheets
- `individual_models/[ModelName]/training_data/`: Individual model training curves

### Raw Research Data
- `research_reconstruction_20250624_152332.json`: Complete reconstruction data
- `research_summary_20250624_152332.json`: Statistical analysis and conclusions

### Visualizations
- `individual_models/[ModelName]/graphs/`: Detailed individual model analysis
- `combined_analysis/`: Comparative visualizations
- `training_graphs/`: All training progression graphs

## How to Reconstruct Analysis

### 1. Load Model for Inference
```python
# YOLOv5
import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path_to_v5.pt')

# YOLOv8/v10/v11
from ultralytics import YOLO
model = YOLO('path_to_model.pt')

# Run inference
results = model('path_to_image.jpg')
```

### 2. Recreate Training Curves
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training data
df = pd.read_csv('combined_training_data_20250624_152332.csv')

# Plot for specific model
model_data = df[df['model'] == 'YOLOv10']
plt.plot(model_data['epoch'], model_data['val_accuracy'])
plt.title('YOLOv10 Validation Accuracy')
plt.show()
```

### 3. Performance Comparison
```python
import json

# Load research summary
with open('research_summary_20250624_152332.json', 'r') as f:
    summary = json.load(f)

# Access performance metrics
ranking = summary['conclusions']['performance_ranking']
best_model = ranking[0]
```

## Research Findings

### Performance Ranking
1. **YOLOv10** - 98.7% accuracy
2. **YOLOv11** - 97.8% accuracy
3. **YOLOv8** - 95.4% accuracy
4. **YOLOv5** - 92.5% accuracy

### Key Insights
- **Best Overall Performance:** YOLOv10
- **Fastest Inference:** YOLOv8
- **Most Efficient:** YOLOv10
- **Smallest Model:** YOLOv10

## Citation

If you use this research data, please cite:
```
Snail Species Detection using YOLO Variants
Generated: 2025-06-24
Research ID: 20250624_152332
```

---
*This research data was generated using an automated YOLO comparison system with authentic model analysis and reconstruction capabilities.*
