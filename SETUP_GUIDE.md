# Photo Metadata AI - Complete Setup Guide

This guide will walk you through setting up the Photo Metadata AI system from scratch, including automatic bucket creation.

## Prerequisites

- Python 3.7+
- AWS Account with S3 and Rekognition access
- OpenAI API key (for AI analysis features)

## Step 1: Install the Package

```bash
# Install from GitHub
pip install git+https://github.com/elephant-xyz/photo-meta-data-ai.git

# Or install locally
git clone https://github.com/elephant-xyz/photo-meta-data-ai.git
cd photo-meta-data-ai
pip install -e .
```

## Step 2: Set Environment Variables

Create a `.env` file in your project directory:

```bash
# AWS Credentials
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=us-east-1

# S3 Configuration
S3_BUCKET_NAME=photo-metadata-ai

# OpenAI API (for AI analysis)
OPENAI_API_KEY=your-openai-api-key

# Optional configurations
IMAGES_DIR=images
OUTPUT_BASE_FOLDER=output
```

Or set them in your shell:

```bash
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
export S3_BUCKET_NAME="photo-metadata-ai"
export OPENAI_API_KEY="your-openai-api-key"
```

## Step 3: Create Your Data Files

### Create `seed.csv`
```csv
parcel_id,Address,method,headers,url,multiValueQueryString,body,json,source_identifier,County
30434108090030050,"1605 S US HIGHWAY 1 3E,PALM BEACH GARDENS","GET",,"https://pbcpao.gov/Property/Details",{"parcelId":["30434108090030050"]},,,30434108090030050,palm beach
52434205310037080,"2558 GARDENS PKWY,JUPITER","GET",,"https://pbcpao.gov/Property/Details",{"parcelId":["52434205310037080"]},,,52434205310037080,palm beach
```

### Create `upload-results.csv` (if using parcel processor)
```csv
propertyCid,dataGroupCid,dataCid,filePath,uploadedAt
bafkreiepon5udb7ekskmmywzlxyn5bkicw33jed63wu2ic3a6qj3bkn4ty,bafkreicejtlqsmjzaz7wo2rfp7wdfihuayyl3x342z3evr46t6qym4h6be,bafkreiepon5udb7ekskmmywzlxyn5bkicw33jed63wu2ic3a6qj3bkn4ty,"/content/output/52434205310037080/bafkreicejtlqsmjzaz7wo2rfp7wdfihuayyl3x342z3evr46t6qym4h6be.json",2025-07-22T14:59:46.733Z
```

## Step 4: Prepare Your Images

Create the folder structure with parcel IDs as folder names:

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

## Step 5: Optional - Create Bucket Manually

The tools will automatically create the S3 bucket if it doesn't exist, but you can also create it manually:

```bash
# Create bucket with default settings
bucket-manager --create

# Create bucket with custom name
bucket-manager --create --bucket-name my-custom-bucket

# Create bucket in specific region
bucket-manager --create --bucket-name my-custom-bucket --region us-west-2

# Check bucket information
bucket-manager --info

# List all available buckets
bucket-manager --list
```

## Step 6: Run Your Workflow

### Option A: Complete Workflow (Recommended)
```bash
# This will upload, categorize, and analyze all properties from seed.csv
photo-categorizer
```

### Option B: Step-by-Step Workflow
```bash
# 1. Upload images to S3
python src/uploadtoS3.py

# 2. Categorize images
photo-categorizer

# 3. Run AI analysis
ai-analyzer --all-properties
```

### Option C: Parcel-Based Workflow
```bash
# Process using parcel IDs and property CIDs from CSV files
parcel-processor
```

## Step 7: Verify Setup

### Check Bucket Contents
```bash
# List all buckets
bucket-manager --list

# Check bucket information
bucket-manager --info
```

### Check Logs
```bash
# View logs for each tool
tail -f logs/photo-categorizer.log
tail -f logs/ai-analyzer.log
tail -f logs/parcel-processor.log
tail -f logs/bucket-manager.log
```

## Expected Output Structure

After running the tools, your S3 bucket will contain:

```
photo-metadata-ai/
├── 30434108090030050/
│   ├── kitchen/
│   │   ├── kitchen1.jpg
│   │   └── categorization_results.json
│   ├── bedroom/
│   │   └── bedroom1.jpg
│   └── bathroom/
│       └── bathroom1.jpg
├── 52434205310037080/
│   ├── exterior/
│   │   └── exterior1.jpg
│   └── garage/
│       └── garage1.jpg
```

And your local `output/` directory will contain:

```
output/
├── 30434108090030050/
│   ├── property.json
│   ├── relationships.json
│   └── [individual object files]
└── 52434205310037080/
    ├── property.json
    ├── relationships.json
    └── [individual object files]
```

## Troubleshooting

### Common Issues

1. **Bucket Creation Fails**
   - Ensure AWS credentials have S3 permissions
   - Check if bucket name is globally unique
   - Verify region settings

2. **Authentication Errors**
   - Verify AWS credentials are correct
   - Check that credentials have necessary permissions
   - Ensure region matches your AWS account

3. **Missing Files**
   - Ensure `seed.csv` exists in current directory
   - Verify `images/` folder structure matches parcel IDs
   - Check that `upload-results.csv` exists (for parcel processor)

4. **OpenAI API Errors**
   - Verify `OPENAI_API_KEY` is set correctly
   - Check API key has sufficient credits
   - Ensure API key has access to GPT-4 Vision

### Debug Mode

To see more detailed logs, you can modify the logging level in any script:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Security Features

The bucket manager automatically configures:

- **Versioning**: Tracks all object versions
- **Encryption**: AES256 server-side encryption
- **Private Access**: Blocks all public access
- **Secure Transport**: Requires HTTPS for all operations
- **Security Policies**: Denies non-HTTPS access

## Cost Optimization

- **S3 Storage**: ~$0.023 per GB/month
- **Rekognition**: ~$1.00 per 1000 images
- **OpenAI API**: ~$0.01-0.03 per image (depending on complexity)
- **Data Transfer**: Free within same region

## Support

For issues or questions:
- Check the logs in the `logs/` directory
- Review the individual tool documentation
- Ensure all environment variables are set correctly 