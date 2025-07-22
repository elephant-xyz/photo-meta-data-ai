#!/usr/bin/env python3
"""
Script to process photos using parcel IDs from CSV data and property CIDs from upload results.
Reads from upload_results.csv and parcel data CSV to coordinate photo processing.
"""

import os
import sys
import csv
import logging
import pandas as pd
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from collections import defaultdict
import time
from datetime import datetime
from dotenv import load_dotenv


class ParcelProcessor:
    def __init__(self):
        self.s3_client = None
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai')
        
        # Load environment variables from .env file
        self.load_environment()
        
        # Setup logging
        self.setup_logging()

    def load_environment(self):
        """Load environment variables from .env file"""
        # Try to load from .env file
        env_paths = [
            '.env',
            '/content/.env',
            os.path.expanduser('~/.env')
        ]
        
        env_loaded = False
        for env_path in env_paths:
            if os.path.exists(env_path):
                load_dotenv(env_path)
                print(f"✓ Loaded environment from {env_path}")
                env_loaded = True
                break
        
        if not env_loaded:
            print("⚠️  No .env file found, using system environment variables")
        
        return env_loaded

    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/parcel-processor.log'),
                logging.StreamHandler()  # Also log to console
            ]
        )
        self.logger = logging.getLogger(__name__)

    def authenticate_aws(self):
        """Authenticate with AWS services"""
        try:
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

            if not aws_access_key or not aws_secret_key:
                self.logger.error("Error: AWS credentials not found in environment variables!")
                self.logger.error("Please set the following environment variables:")
                self.logger.error("- AWS_ACCESS_KEY_ID")
                self.logger.error("- AWS_SECRET_ACCESS_KEY")
                self.logger.error("- AWS_DEFAULT_REGION (optional, defaults to us-east-1)")
                sys.exit(1)

            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )

            # Test connections
            self.s3_client.list_buckets()
            self.logger.info("✓ AWS S3 authentication successful")

        except NoCredentialsError:
            self.logger.error("Error: AWS credentials not found!")
            sys.exit(1)
        except ClientError as e:
            self.logger.error(f"Error: AWS authentication failed - {e}")
            sys.exit(1)

    def load_upload_results(self, upload_results_path):
        """Load upload results CSV to get property CIDs and data CIDs"""
        try:
            df = pd.read_csv(upload_results_path)
            self.logger.info(f"✓ Loaded {len(df)} records from upload_results.csv")
            
            # Create mapping from dataGroupCid to propertyCid
            cid_mapping = {}
            for _, row in df.iterrows():
                data_group_cid = row['dataGroupCid']
                property_cid = row['propertyCid']
                cid_mapping[data_group_cid] = property_cid
                self.logger.debug(f"Mapped {data_group_cid} -> {property_cid}")
            
            self.logger.info(f"✓ Created {len(cid_mapping)} CID mappings")
            return cid_mapping
            
        except Exception as e:
            self.logger.error(f"Error loading upload_results.csv: {e}")
            return {}

    def load_seed_data(self, seed_data_path):
        """Load seed data CSV to get parcel IDs and addresses"""
        try:
            df = pd.read_csv(seed_data_path)
            self.logger.info(f"✓ Loaded {len(df)} records from seed data CSV")
            
            # Create mapping from parcel_id to address
            parcel_mapping = {}
            for _, row in df.iterrows():
                parcel_id = str(row['parcel_id'])
                address = row['Address']
                parcel_mapping[parcel_id] = address
                self.logger.debug(f"Mapped parcel {parcel_id} -> {address}")
            
            self.logger.info(f"✓ Created {len(parcel_mapping)} parcel mappings")
            return parcel_mapping
            
        except Exception as e:
            self.logger.error(f"Error loading seed data CSV: {e}")
            return {}

    def get_property_images(self, parcel_id):
        """Get list of all images for a parcel from S3"""
        try:
            prefix = f"{parcel_id}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                self.logger.warning(f"No images found for parcel {parcel_id}")
                return []

            images = []
            for obj in response['Contents']:
                key = obj['Key']
                if self.is_image_file(key):
                    images.append(key)

            self.logger.info(f"Found {len(images)} images for parcel {parcel_id}")
            return images

        except ClientError as e:
            self.logger.error(f"Error listing images for parcel {parcel_id}: {e}")
            return []

    def is_image_file(self, key):
        """Check if file is an image based on extension"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        return Path(key).suffix.lower() in image_extensions

    def list_parcel_categories(self, parcel_id):
        """List all category folders for a specific parcel in S3"""
        try:
            parcel_prefix = f"{parcel_id}/"
            self.logger.debug(f"Listing category folders for parcel {parcel_id} with prefix: {parcel_prefix}")
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=parcel_prefix,
                Delimiter='/'
            )
            
            self.logger.debug(f"S3 response keys: {list(response.keys())}")
            self.logger.debug(f"CommonPrefixes: {response.get('CommonPrefixes', [])}")
            
            categories = []
            for prefix in response.get('CommonPrefixes', []):
                category_name = prefix['Prefix'].rstrip('/').split('/')[-1]
                categories.append(category_name)
                self.logger.debug(f"Found category: {category_name}")
            
            self.logger.debug(f"Total categories found for {parcel_id}: {len(categories)}")
            return categories
        except Exception as e:
            self.logger.error(f"Error listing S3 category folders for parcel {parcel_id}: {e}")
            return []

    def process_parcel(self, parcel_id, property_cid, address):
        """Process all images for a parcel"""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Processing Parcel: {parcel_id}")
        self.logger.info(f"Property CID: {property_cid}")
        self.logger.info(f"Address: {address}")
        self.logger.info(f"{'=' * 60}")

        # List all category folders for this parcel
        categories = self.list_parcel_categories(parcel_id)
        
        if not categories:
            self.logger.warning(f"⚠️  No category folders found for parcel {parcel_id}, skipping...")
            return False
        
        self.logger.info(f"📁 Found {len(categories)} category folders for {parcel_id}: {', '.join(categories)}")
        
        # Process each category
        total_images = 0
        for category in categories:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🏠 Processing Category: {category}")
            self.logger.info(f"{'='*60}")
            
            # Get images in this category
            images = self.get_property_images(parcel_id)
            category_images = [img for img in images if f"/{category}/" in img]
            
            if category_images:
                self.logger.info(f"Found {len(category_images)} images in category {category}")
                total_images += len(category_images)
            else:
                self.logger.warning(f"No images found in category {category}")
        
        self.logger.info(f"Total images found for parcel {parcel_id}: {total_images}")
        return total_images > 0

    def run_ai_analysis(self, parcel_id, property_cid):
        """Run AI analysis for a parcel using the property CID for relationships"""
        try:
            # Import the AI analyzer
            from .ai_image_analysis_optimized_multi_thread import main as ai_main
            import sys
            
            # Set up arguments for AI analyzer
            sys.argv = [
                'ai-analyzer',
                '--property-id', parcel_id,
                '--output-dir', 'output'
            ]
            
            self.logger.info(f"Starting AI analysis for parcel {parcel_id} with property CID {property_cid}")
            
            # Run the AI analysis
            ai_main()
            
            self.logger.info(f"✓ Completed AI analysis for parcel {parcel_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error running AI analysis for parcel {parcel_id}: {e}")
            return False

    def process_all_parcels(self, upload_results_path, seed_data_path):
        """Process all parcels using the provided CSV files"""
        self.logger.info("Starting parcel processing workflow...")
        
        # Load data from CSV files
        cid_mapping = self.load_upload_results(upload_results_path)
        parcel_mapping = self.load_seed_data(seed_data_path)
        
        if not cid_mapping:
            self.logger.error("Failed to load upload results. Exiting.")
            return False
            
        if not parcel_mapping:
            self.logger.error("Failed to load parcel data. Exiting.")
            return False
        
        # Process each parcel
        total_parcels = 0
        successful_parcels = 0
        
        for parcel_id, address in parcel_mapping.items():
            # Find the corresponding property CID
            # We need to match based on some criteria - for now, let's assume
            # the parcel_id is used as a key in the upload results
            property_cid = None
            
            # Look for matching property CID in upload results
            for data_group_cid, prop_cid in cid_mapping.items():
                # This is a simplified matching - you may need to adjust based on your data structure
                if parcel_id in data_group_cid or data_group_cid in parcel_id:
                    property_cid = prop_cid
                    break
            
            if not property_cid:
                self.logger.warning(f"No property CID found for parcel {parcel_id}")
                continue
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Processing Parcel {total_parcels + 1}: {parcel_id}")
            self.logger.info(f"Property CID: {property_cid}")
            self.logger.info(f"Address: {address}")
            self.logger.info(f"{'='*80}")
            
            # Process the parcel
            if self.process_parcel(parcel_id, property_cid, address):
                # Run AI analysis
                if self.run_ai_analysis(parcel_id, property_cid):
                    successful_parcels += 1
                    self.logger.info(f"✓ Successfully processed parcel {parcel_id}")
                else:
                    self.logger.error(f"❌ AI analysis failed for parcel {parcel_id}")
            else:
                self.logger.warning(f"⚠️  No images found for parcel {parcel_id}")
            
            total_parcels += 1
        
        # Final summary
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"FINAL SUMMARY")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Total parcels processed: {total_parcels}")
        self.logger.info(f"Successful: {successful_parcels}")
        self.logger.info(f"Failed: {total_parcels - successful_parcels}")
        
        return successful_parcels > 0


def main():
    # Setup logging for main function
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/parcel-processor.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Parcel Processor - Photo Processing with Property CIDs")
    logger.info("=" * 60)
    
    # Check for required files
    upload_results_path = "upload_results.csv"
    seed_data_path = "seed.csv"
    
    if not os.path.exists(upload_results_path):
        logger.error(f"❌ {upload_results_path} not found!")
        logger.error("Please provide upload_results.csv with propertyCid,dataGroupCid,dataCid,filePath,uploadedAt columns")
        sys.exit(1)
    
    if not os.path.exists(seed_data_path):
        logger.error(f"❌ {seed_data_path} not found!")
        logger.error("Please provide seed.csv with parcel_id,Address columns")
        sys.exit(1)
    
    logger.info(f"✓ Found {upload_results_path}")
    logger.info(f"✓ Found {seed_data_path}")
    
    # Initialize processor
    processor = ParcelProcessor()
    
    # Authenticate with AWS
    logger.info("\n1. Authenticating with AWS...")
    processor.authenticate_aws()
    
    # Ensure bucket exists
    logger.info("\n2. Ensuring S3 bucket exists...")
    from .bucket_manager import BucketManager
    bucket_manager = BucketManager()
    bucket_manager.authenticate_aws()
    bucket_success = bucket_manager.ensure_bucket_exists()
    if not bucket_success:
        logger.error("\n❌ Bucket setup failed! Cannot proceed.")
        sys.exit(1)
    logger.info("\n✅ Bucket is ready!")
    
    # Process all parcels
    logger.info("\n3. Processing all parcels...")
    success = processor.process_all_parcels(upload_results_path, seed_data_path)
    
    if success:
        logger.info("\n🎉 Parcel processing completed successfully!")
    else:
        logger.error("\n❌ Parcel processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 