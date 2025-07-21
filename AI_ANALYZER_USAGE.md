# AI Image Analyzer - Command Line Usage

## Overview

The AI Image Analyzer processes real estate images using OpenAI's GPT-4 Vision and IPFS schemas to extract detailed property information.

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
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# Optional (defaults shown)
export S3_BUCKET_NAME="photo-metadata-ai"
export OUTPUT_BASE_FOLDER="output"
```

## Usage

### Process All Properties in S3
```bash
python src/ai_analyzer.py --all-properties
```

### Process a Specific Property
```bash
python src/ai_analyzer.py --property-id 30434108090030050
```

### Custom Configuration
```bash
python src/ai_analyzer.py --all-properties \
    --batch-size 10 \
    --max-workers 5 \
    --output-dir /path/to/output \
    --bucket my-custom-bucket
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--property-id` | Process specific property ID | None |
| `--all-properties` | Process all properties in S3 | False |
| `--batch-size` | Number of images per batch | 5 |
| `--max-workers` | Maximum parallel workers | 3 |
| `--output-dir` | Output directory | output |
| `--bucket` | S3 bucket name | S3_BUCKET_NAME env var |

## Examples

### Basic Usage
```bash
# Process all properties
python src/ai_analyzer.py --all-properties

# Process specific property
python src/ai_analyzer.py --property-id 30434108090030050
```

### Advanced Usage
```bash
# High-performance processing
python src/ai_analyzer.py --all-properties \
    --batch-size 15 \
    --max-workers 8 \
    --output-dir /data/analysis

# Custom bucket
python src/ai_analyzer.py --all-properties \
    --bucket my-real-estate-bucket
```

### Environment File
Create `.env` file:
```env
OPENAI_API_KEY=sk-your-openai-key
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=photo-metadata-ai
OUTPUT_BASE_FOLDER=output
```

## Output Structure

The analyzer creates detailed JSON files for each property:

```
output/
├── 30434108090030050/
│   ├── kitchen/
│   │   ├── appliances.json
│   │   ├── layout.json
│   │   └── relationships.json
│   ├── bedroom/
│   │   ├── layout.json
│   │   └── relationships.json
│   └── bathroom/
│       ├── fixtures.json
│       └── relationships.json
└── 30434108090030051/
    └── ...
```

## Features

- 🔍 **AI-Powered Analysis**: Uses OpenAI GPT-4 Vision
- 📊 **IPFS Schemas**: Structured data extraction
- 🏠 **Real Estate Focused**: Property-specific categories
- ⚡ **Parallel Processing**: Multi-threaded for speed
- 📁 **S3 Integration**: Direct S3 bucket processing
- 🔗 **Relationship Mapping**: Creates property relationships

## Requirements

- Python 3.7+
- OpenAI API key
- AWS credentials with S3 access
- Images already categorized in S3 (use photo-categorizer first)

## Error Handling

The tool includes comprehensive error handling:
- ✅ Environment validation
- ✅ AWS authentication
- ✅ S3 bucket access
- ✅ OpenAI API connectivity
- ✅ Graceful interruption handling 