# Parallel Category Processor - Usage Guide

## Overview

The Parallel Category Processor processes images from category folders in parallel, with automatic batching when there are more than 10 images per category. Different categories are processed simultaneously for maximum efficiency.

## Features

- **Parallel Processing**: Different categories are processed simultaneously
- **Automatic Batching**: Images are batched when there are more than 10 per category
- **Cost Tracking**: Real-time cost monitoring and reporting
- **Error Handling**: Robust error handling with retry logic
- **Logging**: Comprehensive logging for debugging and monitoring

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file or set environment variables:

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage

### Process a Single Property
```bash
python src/parallel_category_processor.py 30434108090030050
```

### Process Multiple Properties
```bash
python src/parallel_category_processor.py 30434108090030050 52434205310037080
```

### Process All Properties
```bash
python src/parallel_category_processor.py --all
```

## How It Works

### 1. Category Discovery
The script automatically discovers category folders within each property:
- `images/30434108090030050/bedroom/`
- `images/30434108090030050/kitchen/`
- `images/30434108090030050/exterior/`
- etc.

### 2. Parallel Processing
Each category is processed in parallel using ThreadPoolExecutor:
- Up to 5 categories processed simultaneously (configurable)
- Each category runs independently
- No dependencies between categories

### 3. Batching Logic
- **≤ 10 images**: Processed in a single batch
- **> 10 images**: Automatically split into batches of 10

### 4. Output Structure
```
output/
├── 30434108090030050/
│   ├── bedroom_batch_01.json
│   ├── kitchen_batch_01.json
│   ├── kitchen_batch_02.json
│   └── exterior_batch_01.json
└── 52434205310037080/
    ├── bedroom_batch_01.json
    └── kitchen_batch_01.json
```

## Configuration

### Batch Size
```python
BATCH_SIZE = 10  # Maximum images per batch
```

### Parallel Workers
```python
MAX_WORKERS = 5  # Maximum parallel workers for categories
```

### Image Optimization
```python
MAX_IMAGE_SIZE = (1024, 1024)  # Maximum image dimensions
JPEG_QUALITY = 85  # JPEG compression quality
```

## Example Output

### Single Batch (≤ 10 images)
```json
{
  "category": "bedroom",
  "images_analyzed": 4,
  "features": [
    {
      "feature_name": "Queen Bed",
      "description": "Modern queen-size bed with upholstered headboard",
      "condition": "Excellent",
      "quality_rating": 9
    }
  ],
  "summary": "Well-maintained bedroom with modern furnishings"
}
```

### Multiple Batches (> 10 images)
- `kitchen_batch_01.json` (first 10 images)
- `kitchen_batch_02.json` (next 10 images)
- etc.

## Performance Benefits

1. **Parallel Processing**: Categories don't wait for each other
2. **Efficient Batching**: Optimal batch sizes for API limits
3. **Cost Optimization**: Balanced between speed and cost
4. **Resource Management**: Thread-safe operations

## Monitoring

### Logs
- All operations logged to `logs/parallel_category_processor.log`
- Real-time console output
- Error tracking and reporting

### Metrics
- Total images processed
- Total cost incurred
- Processing time
- Success/failure rates

## Error Handling

- **API Failures**: Automatic retry with exponential backoff
- **Image Errors**: Skip invalid images, continue processing
- **Timeout Protection**: 10-minute timeout per category
- **Graceful Degradation**: Continue processing other categories if one fails

## Best Practices

1. **Test with Small Properties**: Start with properties that have few images
2. **Monitor Costs**: Check cost estimates before large runs
3. **Check Logs**: Review logs for any issues
4. **Backup Data**: Ensure important data is backed up

## Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   ```
   ❌ OPENAI_API_KEY not found in environment variables
   ```
   Solution: Set the environment variable

2. **Property Folder Not Found**
   ```
   Property folder not found: images/123456789
   ```
   Solution: Check the property ID and folder structure

3. **No Images Found**
   ```
   No images found in bedroom
   ```
   Solution: Check if the category folder contains image files

4. **API Rate Limits**
   ```
   OpenAI API call failed: Rate limit exceeded
   ```
   Solution: Reduce MAX_WORKERS or add delays between requests

### Performance Tuning

- **Increase Speed**: Increase MAX_WORKERS (be careful with rate limits)
- **Reduce Cost**: Decrease BATCH_SIZE
- **Improve Quality**: Increase MAX_IMAGE_SIZE and JPEG_QUALITY
- **Handle Large Properties**: Increase timeout values

## Integration

This script can be integrated with existing workflows:

```python
from src.parallel_category_processor import process_property_parallel

# Process a property programmatically
cost = process_property_parallel("30434108090030050")
print(f"Processing cost: ${cost:.4f}")
``` 