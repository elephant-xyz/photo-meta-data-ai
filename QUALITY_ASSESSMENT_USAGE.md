# Image Quality Assessment and Duplicate Detection

This feature enhances the AI analysis by adding image quality assessment and duplicate detection capabilities.

## Features

### 🎯 **Quality Assessment**
- **Resolution scoring**: Evaluates image resolution (HD, SD, etc.)
- **Brightness analysis**: Checks for proper exposure levels
- **Contrast measurement**: Assesses image contrast quality
- **Sharpness detection**: Uses Laplacian variance for sharpness
- **File size analysis**: Tracks file size for storage optimization
- **Overall quality rating**: 0-10 score with descriptive ratings

### 🔍 **Duplicate Detection**
- **Perceptual hashing**: Fast exact duplicate detection
- **Feature matching**: SIFT-based similarity detection
- **Configurable threshold**: Adjustable similarity sensitivity
- **Duplicate grouping**: Identifies all related duplicates

### 📊 **Quality Mapping JSON**
The system creates a `quality_mapping.json` file with:
```json
{
  "metadata": {
    "total_images": 150,
    "created_at": "2024-01-15 14:30:00",
    "version": "1.0"
  },
  "images": {
    "image_001.jpg": {
      "file_path": "/path/to/image_001.jpg",
      "quality_assessment": {
        "quality_score": 8,
        "quality_rating": "excellent",
        "resolution": "1920x1080",
        "aspect_ratio": 1.78,
        "brightness": 127.5,
        "contrast": 45.2,
        "sharpness": 156.8,
        "file_size_mb": 2.4
      },
      "is_duplicate": false,
      "duplicate_of": [],
      "duplicate_count": 0
    }
  }
}
```

## Installation

1. **Install additional dependencies:**
   ```bash
   pip install -r requirements_quality.txt
   ```

2. **For Google Colab:**
   ```python
   !pip install opencv-python numpy scikit-image scikit-learn imagehash
   ```

## Usage

### **Command Line Usage**
```bash
# Basic quality assessment
python src/ai_image_analysis_with_quality_duplicate_detection.py --input-dir /path/to/images --output-dir output

# With custom similarity threshold
python src/ai_image_analysis_with_quality_duplicate_detection.py --input-dir /path/to/images --output-dir output --similarity-threshold 0.7
```

### **Google Colab Usage**
```python
# Upload the script
from google.colab import files
uploaded = files.upload()  # Select ai_image_analysis_with_quality_duplicate_detection.py

# Run quality assessment
!python ai_image_analysis_with_quality_duplicate_detection.py --input-dir /content/images --output-dir /content/output
```

## Quality Ratings

| Score | Rating | Description |
|-------|--------|-------------|
| 8-10 | excellent | High resolution, good exposure, sharp |
| 6-7 | good | Decent quality, minor issues |
| 4-5 | fair | Acceptable but with noticeable issues |
| 2-3 | poor | Low quality, significant problems |
| 0-1 | very_poor | Very low quality or corrupted |

## Quality Metrics

### **Resolution Scoring (0-3 points)**
- **3 points**: 1920x1080 or higher (Full HD+)
- **2 points**: 1280x720 or higher (HD)
- **1 point**: 640x480 or higher (SD)

### **Brightness Scoring (0-2 points)**
- **2 points**: 50-200 (optimal range)
- **1 point**: 30-220 (acceptable range)

### **Contrast Scoring (0-2 points)**
- **2 points**: ≥50 (high contrast)
- **1 point**: ≥30 (moderate contrast)

### **Sharpness Scoring (0-3 points)**
- **3 points**: ≥100 (very sharp)
- **2 points**: ≥50 (sharp)
- **1 point**: ≥20 (somewhat sharp)

## Duplicate Detection

### **Similarity Thresholds**
- **0.9-1.0**: Very strict (near identical)
- **0.8-0.9**: Strict (highly similar)
- **0.7-0.8**: Moderate (similar)
- **0.6-0.7**: Loose (somewhat similar)

### **Detection Methods**
1. **Perceptual Hashing**: Fast exact duplicate detection
2. **SIFT Feature Matching**: Detailed similarity analysis
3. **Combined Approach**: Both methods for comprehensive detection

## Integration with AI Analysis

The quality assessment can be integrated with your existing AI analysis:

```python
# Filter by quality before AI analysis
high_quality_images = []
for img_name, img_data in quality_mapping["images"].items():
    if img_data["quality_assessment"]["quality_score"] >= 6:
        high_quality_images.append(img_data["file_path"])

# Use only high-quality, non-duplicate images
analysis_images = []
for img_path in high_quality_images:
    img_name = os.path.basename(img_path)
    if not quality_mapping["images"][img_name]["is_duplicate"]:
        analysis_images.append(img_path)
```

## Output Files

1. **`quality_mapping.json`**: Complete quality and duplicate data
2. **Quality summary**: Printed to console with statistics
3. **Filtered image lists**: Ready for AI analysis

## Benefits

- ✅ **Reduce processing time** by filtering out duplicates
- ✅ **Improve AI accuracy** by using only high-quality images
- ✅ **Optimize storage** by identifying redundant images
- ✅ **Quality control** for consistent results
- ✅ **Detailed metrics** for image optimization

## Example Output

```
============================================================
IMAGE QUALITY SUMMARY
============================================================
Total Images: 150
Duplicates: 12

Quality Distribution:
  excellent: 45 (30.0%)
  good: 67 (44.7%)
  fair: 28 (18.7%)
  poor: 8 (5.3%)
  very_poor: 2 (1.3%)

Non-duplicate images for AI analysis: 138
```

This enhancement will significantly improve your AI analysis pipeline by ensuring only the best quality, unique images are processed! 