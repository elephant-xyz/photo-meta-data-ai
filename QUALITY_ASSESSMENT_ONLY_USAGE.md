# Quality Assessment and Duplicate Detection Script

This script performs quality assessment and duplicate detection on images stored in S3, without running AI analysis.

## Features

- **Image Quality Assessment**: Evaluates resolution, brightness, contrast, and sharpness
- **Duplicate Detection**: Uses perceptual hashing and SIFT feature matching
- **S3 Integration**: Works directly with your S3 bucket
- **Detailed Reports**: Generates comprehensive quality mapping JSON files

## Installation

1. Install dependencies:
```bash
pip install -r requirements_quality_only.txt
```

2. Ensure your `.env` file contains:
```
S3_BUCKET_NAME=your-bucket-name
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=your-region
```

## Usage

### Process a Specific Folder
```bash
python src/quality_assessment_only.py --folder "52434205310037080/dining_room"
```

### Process All Folders
```bash
python src/quality_assessment_only.py --all-folders
```

### Custom Similarity Threshold
```bash
python src/quality_assessment_only.py --folder "52434205310037080/dining_room" --similarity-threshold 0.7
```

### Custom Output Directory
```bash
python src/quality_assessment_only.py --folder "52434205310037080/dining_room" --output-dir "custom_output"
```

## Output

The script creates a `quality_mapping.json` file in the output directory with:

### Quality Metrics
- **Quality Score**: 0-10 rating based on multiple factors
- **Quality Rating**: excellent/good/fair/poor/very_poor
- **Resolution**: Image dimensions
- **Brightness**: Average pixel brightness
- **Contrast**: Standard deviation of pixel values
- **Sharpness**: Laplacian variance measure
- **File Size**: Size in MB

### Duplicate Detection
- **Is Duplicate**: Boolean flag
- **Duplicate Of**: List of duplicate image names
- **Duplicate Count**: Number of duplicates found

## Quality Scoring System

- **Resolution (0-3 points)**: 1920x1080+ (3), 1280x720+ (2), 640x480+ (1)
- **Brightness (0-2 points)**: 50-200 (2), 30-220 (1)
- **Contrast (0-2 points)**: ≥50 (2), ≥30 (1)
- **Sharpness (0-3 points)**: ≥100 (3), ≥50 (2), ≥20 (1)

## Example Output

```json
{
  "metadata": {
    "total_images": 15,
    "created_at": "2024-01-15 14:30:25",
    "similarity_threshold": 0.8,
    "version": "1.0"
  },
  "images": {
    "image1.jpg": {
      "file_path": "/tmp/temp_image1.jpg",
      "quality_assessment": {
        "quality_score": 8,
        "quality_rating": "excellent",
        "resolution": "1920x1080",
        "aspect_ratio": 1.78,
        "brightness": 127.5,
        "contrast": 65.2,
        "sharpness": 125.8,
        "file_size_mb": 2.45
      },
      "is_duplicate": false,
      "duplicate_of": [],
      "duplicate_count": 0
    }
  }
}
```

## Logging

Logs are saved to `logs/quality-assessment.log` with detailed information about:
- Processing progress
- Quality assessment results
- Duplicate detection findings
- Error messages

## Performance

- Processes images in parallel for faster assessment
- Downloads images temporarily for analysis
- Cleans up temporary files automatically
- Memory efficient for large image sets

## Troubleshooting

1. **AWS Authentication Error**: Check your AWS credentials in `.env`
2. **S3 Bucket Not Found**: Verify bucket name in environment variables
3. **OpenCV Error**: Ensure opencv-python is properly installed
4. **Memory Issues**: Process smaller batches for large image sets 