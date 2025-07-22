# Parcel Processor Usage Guide

The `parcel-processor` is a new tool that coordinates photo processing using parcel IDs from CSV data and property CIDs from upload results.

## Overview

This tool reads from two CSV files:
1. **`upload_results.csv`** - Contains property CIDs and data CIDs from uploads
2. **`seed.csv`** - Contains parcel IDs and addresses

It then processes photos using parcel IDs and uses property CIDs for relationships.

## Required Files

### 1. upload_results.csv
```csv
propertyCid,dataGroupCid,dataCid,filePath,uploadedAt
bafkreiepon5udb7ekskmmywzlxyn5bkicw33jed63wu2ic3a6qj3bkn4ty,bafkreicejtlqsmjzaz7wo2rfp7wdfihuayyl3x342z3evr46t6qym4h6be,bafkreiepon5udb7ekskmmywzlxyn5bkicw33jed63wu2ic3a6qj3bkn4ty,"/content/output/52434205310037080/bafkreicejtlqsmjzaz7wo2rfp7wdfihuayyl3x342z3evr46t6qym4h6be.json",2025-07-22T14:59:46.733Z
```

### 2. seed.csv
```csv
parcel_id,Address,method,headers,url,multiValueQueryString,body,json,source_identifier,County
30434108090030050,"1605 S US HIGHWAY 1 3E,PALM BEACH GARDENS","GET",,"https://pbcpao.gov/Property/Details",{"parcelId":["30434108090030050"]},,,30434108090030050,palm beach
```

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/elephant-xyz/photo-meta-data-ai.git

# Or install locally
pip install -e .
```

## Environment Variables

Set these environment variables:
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
export S3_BUCKET_NAME="photo-metadata-ai"
export OPENAI_API_KEY="your-openai-key"
```

## Usage

### Basic Usage
```bash
# Place your CSV files in the current directory
# Then run the parcel processor
parcel-processor
```

### Expected Workflow

1. **Photos are uploaded and categorized** using `photo-categorizer`
   - Photos are organized as: `s3://bucket/parcel-id/category/image.jpg`

2. **Upload results are saved** to `upload_results.csv`
   - Contains property CIDs and data CIDs

3. **Seed data is provided** in `seed.csv`
   - Contains parcel IDs and addresses

4. **Parcel processor runs** and:
   - Reads both CSV files
   - Maps parcel IDs to property CIDs
   - Processes photos using parcel IDs
   - Uses property CIDs for relationships

## Logging

Logs are written to:
- Console output
- `logs/parcel-processor.log`

## Example Output

```
Parcel Processor - Photo Processing with Property CIDs
============================================================
✓ Found upload_results.csv
✓ Found parcel_data.csv

1. Authenticating with AWS...
✓ AWS S3 authentication successful

2. Processing all parcels...
✓ Loaded 2 records from upload_results.csv
✓ Created 2 CID mappings
✓ Loaded 2 records from parcel data CSV
✓ Created 2 parcel mappings

================================================================================
Processing Parcel 1: 30434108090030050
Property CID: bafkreif6myl6xdmu6nnm52xeafzd5jtcppzxqruq73ysax5hqexrccxkbe
Address: 1605 S US HIGHWAY 1 3E,PALM BEACH GARDENS
================================================================================

============================================================
Processing Parcel: 30434108090030050
Property CID: bafkreif6myl6xdmu6nnm52xeafzd5jtcppzxqruq73ysax5hqexrccxkbe
Address: 1605 S US HIGHWAY 1 3E,PALM BEACH GARDENS
============================================================
📁 Found 3 category folders for 30434108090030050: kitchen, bedroom, bathroom

============================================================
🏠 Processing Category: kitchen
============================================================
Found 5 images in category kitchen

============================================================
🏠 Processing Category: bedroom
============================================================
Found 3 images in category bedroom

============================================================
🏠 Processing Category: bathroom
============================================================
Found 2 images in category bathroom
Total images found for parcel 30434108090030050: 10

Starting AI analysis for parcel 30434108090030050 with property CID bafkreif6myl6xdmu6nnm52xeafzd5jtcppzxqruq73ysax5hqexrccxkbe
✓ Completed AI analysis for parcel 30434108090030050
✓ Successfully processed parcel 30434108090030050

======================================================================
FINAL SUMMARY
======================================================================
Total parcels processed: 2
Successful: 2
Failed: 0

🎉 Parcel processing completed successfully!
```

## Key Features

- **Parcel-based processing**: Uses parcel IDs for photo organization
- **Property CID relationships**: Uses property CIDs from upload results for relationships
- **CSV-driven workflow**: Reads from structured CSV files
- **Comprehensive logging**: Detailed logs for debugging
- **Error handling**: Graceful handling of missing data or failed operations

## File Structure

After processing, you'll have:
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

1. **Missing CSV files**: Ensure both `upload_results.csv` and `seed.csv` exist
2. **No images found**: Check that photos were uploaded with parcel IDs as folder names
3. **CID mapping issues**: Verify that parcel IDs in CSV match the folder structure in S3
4. **AWS authentication**: Ensure AWS credentials are properly set

### Debug Mode

To see more detailed logs, you can modify the logging level in the script:
```python
logging.basicConfig(level=logging.DEBUG, ...)
``` 