# 🏠 Photo Metadata AI - AWS Rekognition Photo Categorizer

## 📋 Overview

This tool automatically analyzes and categorizes real estate photos using AWS Rekognition AI. It uploads images from your local folders to S3, then uses AI to detect objects and scenes, organizing them into categories like kitchen, bedroom, bathroom, etc.

## 🚀 Quick Start

### 1. Install the Tool
```bash
!pip install git+https://github.com/yourusername/photo-meta-data-ai.git
```

### 2. Set AWS Credentials
```python
import os

# Set your AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'your-access-key-here'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-secret-key-here'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['S3_BUCKET_NAME'] = 'your-bucket-name-here'
```

### 3. Prepare Your Images
Create this folder structure in your Colab environment:
```
images/
├── property-123/
│   ├── kitchen1.jpg
│   ├── bedroom1.jpg
│   ├── bathroom1.jpg
│   └── living1.jpg
├── property-456/
│   ├── exterior1.jpg
│   ├── garage1.jpg
│   └── pool1.jpg
└── property-789/
    ├── office1.jpg
    └── dining1.jpg
```

### 4. Run the Categorizer
```bash
!photo-categorizer
```

## 🎯 What It Does

### Upload Process
- 📤 Uploads images from local `images/` folder to your S3 bucket
- 🗂️ Organizes by property ID: `s3://bucket/property-123/image.jpg`

### AI Analysis
- 🔍 Uses AWS Rekognition to detect objects and scenes
- 📊 Provides confidence scores for each detection
- 🏷️ Identifies furniture, appliances, rooms, and architectural features

### Categorization
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

### Final Organization
```
s3://your-bucket/
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

## 🔧 Usage Options

When you run `photo-categorizer`, you'll get three options:

1. **📤 Upload Only**: Upload images from local folder to S3
2. **🔍 Categorize Only**: Process existing images in S3  
3. **🚀 Upload + Categorize**: Complete workflow (recommended)

## 📊 Results

- ✅ **Organized Images**: Sorted into category folders
- 📈 **JSON Reports**: Detailed analysis with confidence scores
- 📋 **Summary**: Breakdown of images by category
- 🔍 **Labels**: Top detected objects for each image

## 🛠️ Requirements

- ✅ AWS Account with S3 and Rekognition access
- ✅ AWS credentials configured
- ✅ S3 bucket created
- ✅ Images in proper folder structure

## 💡 Tips for Colab

### Upload Images to Colab
```python
from google.colab import files
import os

# Create images directory
!mkdir -p images/property-123

# Upload files (run this cell and select your images)
uploaded = files.upload()

# Move uploaded files to proper structure
for filename in uploaded.keys():
    !mv "{filename}" "images/property-123/{filename}"
```

### Check Results
```python
# List categorized images in S3
import boto3

s3 = boto3.client('s3')
response = s3.list_objects_v2(
    Bucket='your-bucket-name',
    Prefix='property-123/'
)

for obj in response['Contents']:
    print(obj['Key'])
```

## 🔐 Security Notes

- ⚠️ Never commit AWS credentials to version control
- 🔒 Use IAM roles with minimal required permissions
- 📝 Consider using AWS Secrets Manager for production

## 🎉 Example Output

```
AWS Rekognition Photo Categorizer
=============================================
Target S3 Bucket: your-bucket-name
=============================================

1. Authenticating with AWS...
✓ AWS S3 authentication successful
✓ AWS Rekognition client initialized
✓ Using S3 bucket: your-bucket-name

2. Upload Options:
1. Upload images from local folder to S3
2. Process existing images in S3
3. Upload and then process
Choose option (1/2/3): 3

📤 Uploading images to S3...
✓ Found 3 property folders
Processing Property 1/3: property-123
✓ Found 4 image files to upload
[1/4] Processing: kitchen1.jpg
✓ Successfully uploaded kitchen1.jpg
[2/4] Processing: bedroom1.jpg
✓ Successfully uploaded bedroom1.jpg
✅ Upload completed! Proceeding with categorization...

3. Select property to process:
Enter property ID (or 'all' to process all properties): all

4. Starting image analysis and categorization...
Processing Property: property-123
Found 4 images to process

[1/4] Processing: kitchen1.jpg
  Analyzing with Rekognition...
  Top labels: Kitchen (95.2%), Refrigerator (87.1%), Cabinet (82.3%)
  Category: kitchen
  ✓ Successfully categorized as kitchen

🎉 Image categorization completed successfully!
```

## 📞 Support

For issues or questions:
- 📧 Create an issue on GitHub
- 📚 Check the README for detailed documentation
- 🔧 Ensure AWS credentials are properly configured 