# Photo Metadata AI

AWS Rekognition photo categorization tool for real estate images. Automatically analyzes and categorizes property photos into organized folders based on detected content.

## Features

- 🔍 **AI-Powered Analysis**: Uses AWS Rekognition to detect objects and scenes in images
- 🏠 **Real Estate Focused**: Pre-configured categories for kitchen, bedroom, bathroom, living room, etc.
- 📁 **Automatic Organization**: Creates organized folder structure in S3
- 📊 **Detailed Results**: Saves categorization results as JSON with confidence scores
- 🚀 **Batch Processing**: Process single properties or all properties at once

## Quick Start

### Install from GitHub (Recommended)

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
pip install -e .
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Run the categorizer
photo-categorizer
```

## Usage

### Set AWS Credentials

```bash
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
export AWS_DEFAULT_REGION='us-east-1'
export S3_BUCKET_NAME='your-bucket-name'
```

### Run the Categorizer

```bash
# Process a specific property
photo-categorizer

# Or run the script directly
python src/rek.py
```

## How It Works

1. **Authentication**: Connects to AWS S3 and Rekognition services
2. **Image Discovery**: Finds all images for the specified property in S3
3. **AI Analysis**: Uses AWS Rekognition to detect objects and scenes
4. **Categorization**: Maps detected labels to real estate categories
5. **Organization**: Copies images to categorized folders in S3
6. **Results**: Saves detailed categorization results as JSON

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
photo-metadata-ai/
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

## Configuration

### Environment Variables

- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_DEFAULT_REGION`: AWS region (default: us-east-1)
- `S3_BUCKET_NAME`: Your S3 bucket name (default: photo-metadata-ai)

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