# Photo Metadata AI

AWS Rekognition photo categorization tool for real estate images. Automatically analyzes and categorizes property photos into organized folders based on detected content.

## Features

- 🔍 **AI-Powered Analysis**: Uses AWS Rekognition to detect objects and scenes in images
- 🏠 **Real Estate Focused**: Pre-configured categories for kitchen, bedroom, bathroom, living room, etc.
- 📁 **Automatic Organization**: Creates organized folder structure in S3
- 📊 **Detailed Results**: Saves categorization results as JSON with confidence scores
- 🚀 **Batch Processing**: Process single properties or all properties at once
- 📤 **Local to S3 Upload**: Upload images from local folders to S3 before processing
- 🤖 **Advanced AI Analysis**: Deep property analysis using OpenAI GPT-4 Vision and IPFS schemas

## Quick Start

### Remote Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/photo-meta-data-ai.git
cd photo-meta-data-ai

# Quick install with automatic script
./install.sh

# Or install manually
pip install -r requirements.txt
pip install -e .
```

### Install from GitHub

```bash
# Install directly from GitHub repository
pip install git+https://github.com/yourusername/photo-meta-data-ai.git

# Set AWS credentials
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
export AWS_DEFAULT_REGION='us-east-1'
export S3_BUCKET_NAME='your-bucket-name'

# Run the categorizer
photo-categorizer
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/photo-meta-data-ai.git
cd photo-meta-data-ai

# Install with one command
./install.sh

# Or install manually
uv sync

```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# OR
uv sync

# Run the categorizer
photo-categorizer
```

## Available Commands

After installation, you can run any of these commands from anywhere:

### Core AI Analysis
- `ai-analyzer` - Main AI image analysis with county data integration
- `ai-analyzer-quality` - AI analysis with quality and duplicate detection
- `photo-categorizer` - AWS Rekognition photo categorization

### Data Processing
- `fix-schema-validation` - Fix schema validation issues in AI output
- `property-summarizer` - Summarize property data

### File Management
- `copy-property-files-from-zip` - Copy property files from zip
- `copy-all-files-from-zip` - Copy all files from zip
- `unzip-count-data` - Unzip and rename county data
- `copy-all-data-for-submission` - Copy all data for submission

### Setup and Utilities
- `bucket-manager` - S3 bucket management
- `colab-folder-setup` - Colab folder setup
- `quality-assessment` - Image quality assessment
- `upload-to-s3` - Upload files to S3

## Usage

### Set AWS Credentials

```bash
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
export AWS_DEFAULT_REGION='us-east-1'
export S3_BUCKET_NAME='your-bucket-name'
```

### Recent Improvements

**Speed and Performance Optimizations:**

- ✅ Process 3 images per batch with parallel processing (3 workers)
- ✅ Reduced timeout to 180s maximum per API call for faster processing
- ✅ Added 30-minute overall timeout for entire processing
- ✅ Better error handling to continue processing even if batches fail
- ✅ Improved retry logic with exponential backoff
- ✅ All logs now go to files only (no console output)
- ✅ **NEW: WebP image optimization for IPFS** - Automatic conversion to WebP format:
  - All images in output directories are converted to .webp format
  - JSON metadata files automatically reference the .webp filenames
  - Up to 90% file size reduction for efficient IPFS uploads
  - Maintains high image quality (90% quality setting)
  - Original images remain unchanged; only output copies are converted

**Logging and Output:**
- ✅ All scripts use proper logging instead of print statements
- ✅ Logs are written to `logs/` directory with separate files per script
- ✅ No console output - all information goes to log files
- ✅ Proper log levels (INFO, WARNING, ERROR, DEBUG) for better debugging

### Prepare Your Data

1. **Create a `seed.csv` file** with parcel IDs and addresses:

```csv
parcel_id,Address,method,headers,url,multiValueQueryString,body,json,source_identifier,County
30434108090030050,"1605 S US HIGHWAY 1 3E,PALM BEACH GARDENS","GET",,"https://pbcpao.gov/Property/Details",{"parcelId":["30434108090030050"]},,,30434108090030050,palm beach
52434205310037080,"2558 GARDENS PKWY,JUPITER","GET",,"https://pbcpao.gov/Property/Details",{"parcelId":["52434205310037080"]},,,52434205310037080,palm beach
```

2. **Create a folder structure** with parcel IDs as folder names:

```
images/
├── 30434108090030050/
│   ├── kitchen1.jpg
│   ├── bedroom1.jpg
│   ├── bathroom1.jpg
│   └── living1.jpg
├── 52434205310037080/
│   ├── exterior1.jpg
│   ├── garage1.jpg
│   └── pool1.jpg
```

### Run the Categorizer

```bash
# Process a specific property
photo-categorizer

# Or run the script directly
python src/rekognition.py
```

### Run the AI Analyzer

```bash
# Process all properties with AI analysis
ai-analyzer --all-properties

# Process a specific property
ai-analyzer --property-id 30434108090030050

# Advanced usage
ai-analyzer --all-properties --batch-size 10 --max-workers 5
```

### Run the Bucket Manager

```bash
# Create bucket if it doesn't exist
bucket-manager --create

# Show bucket information
bucket-manager --info

# List all available buckets
bucket-manager --list

# Create bucket with custom name and region
bucket-manager --create --bucket-name my-custom-bucket --region us-west-2
```

The bucket manager automatically configures:

- Versioning
- Encryption
- Private access
- Security policies

The tool will give you three options:

1. **Upload images from local folder to S3** - Only upload, don't categorize
2. **Process existing images in S3** - Only categorize, don't upload
3. **Upload and then process** - Upload images and then categorize them

## How It Works

1. **Upload (Optional)**: Uploads images from local `images/` folder to S3
2. **Authentication**: Connects to AWS S3 and Rekognition services
3. **Image Discovery**: Finds all images for the specified property in S3
4. **AI Analysis**: Uses AWS Rekognition to detect objects and scenes
5. **Categorization**: Maps detected labels to real estate categories
6. **Organization**: Copies images to categorized folders in S3
7. **Results**: Saves detailed categorization results as JSON

## Categories

The tool automatically categorizes images into these real estate categories:

- 🍳 **Kitchen**: Appliances, cabinets, countertops
- 🛏️ **Bedroom**: Beds, furniture, sleeping areas
- 🚿 **Bathroom**: Toilets, showers, sinks, mirrors
- 🛋️ **Living Room**: Sofas, TVs, fireplaces
- 🍽️ **Dining Room**: Dining tables, chairs
- 🏠 **Exterior**: Building exteriors, architecture
- 🚗 **Garage**: Cars, vehicles, parking
- 💼 **Office**: Desks, computers, work areas
- 👕 **Laundry**: Washing machines, dryers
- 🪜 **Stairs**: Staircases, railings
- 👔 **Closet**: Wardrobes, clothing storage
- 🏊 **Pool**: Swimming pools, water features
- 🌿 **Balcony**: Terraces, patios, decks
- 📦 **Other**: Unmatched items

## S3 Structure

After processing, your S3 bucket will be organized like this:

```
your-bucket-name/
├── property-123/
│   ├── kitchen/
│   │   ├── kitchen1.jpg
│   │   └── kitchen2.jpg
│   ├── bedroom/
│   │   ├── bedroom1.jpg
│   │   └── bedroom2.jpg
│   ├── bathroom/
│   │   └── bathroom1.jpg
│   └── categorization_results.json
└── property-456/
    ├── living_room/
    │   └── living1.jpg
    └── categorization_results.json
```

## Requirements

- Python 3.7+
- AWS Account with S3 and Rekognition access
- AWS credentials configured
- S3 bucket: Configurable via `S3_BUCKET_NAME` environment variable (default: `photo-metadata-ai`)
- OpenAI API key (for AI analyzer)

## Configuration

### Environment Variables

- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_DEFAULT_REGION`: AWS region (default: us-east-1)
- `S3_BUCKET_NAME`: Your S3 bucket name (default: photo-metadata-ai) - **Automatically created if it doesn't exist**
- `OPENAI_API_KEY`: Your OpenAI API key (for AI analyzer)

### Local Folder Structure

The tool expects images to be stored locally with parcel IDs as folder names:

```
images/
├── 30434108090030050/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── 52434205310037080/
    ├── image1.jpg
    └── ...
```

**Note**: All tools now read from `seed.csv` to determine which properties to process.

### S3 Bucket

The tool expects images to be stored in S3 with this structure:

```
your-bucket-name/
├── property-id-1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── property-id-2/
    ├── image1.jpg
    └── ...
```

## License

MIT License

Perfect! Now the tool is completely automatic. Here's the updated Google Colab code:

```python
# Install the tool from GitHub
!pip install git+https://github.com/elephant-xyz/photo-meta-data-ai.git

# Set AWS credentials
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'your-access-key-here'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-secret-key-here'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['S3_BUCKET_NAME'] = 'your-bucket-name-here'

# Run the photo categorizer (completely automatic)
!photo-categorizer
```

**What it will do automatically:**

1. 📤 **Upload**: All images from `images/` folder to S3
2. 🔍 **Analyze**: Every image with AWS Rekognition
3. 🏷️ **Categorize**: All images into appropriate folders
4. 📊 **Save Results**: JSON reports for each property
5. 📊 **Summary**: Final report of all processed properties

**No user prompts!** The tool will:

- Automatically find all property folders in the images directory
- Upload everything to S3
- Process all properties without asking which ones
- Show progress for each property
- Give you a final summary

**Just run the code and it will process everything in your images folder automatically!**

Make sure your images are organized like this:

```
images/
├── property-123/
│   ├── kitchen1.jpg
│   ├── bedroom1.jpg
│   └── bathroom1.jpg
├── property-456/
│   ├── exterior1.jpg
│   └── garage1.jpg
└── property-789/
    ├── office1.jpg
    └── dining1.jpg
```

