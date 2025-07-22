#!/usr/bin/env python3
"""
Script to upload property images from local folder to AWS S3
Handles folder structure: images/PROPERTY_ID/<all images>
Automatically uploads all properties listed in seed.csv to photo-metadata-ai bucket
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import mimetypes


class PropertyImagesUploader:
    def __init__(self):
        self.s3_client = None
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai').rstrip('/')
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/upload-to-s3.log'),
                logging.StreamHandler()  # Also log to console
            ]
        )
        self.logger = logging.getLogger(__name__)

    def authenticate_aws(self):
        """Authenticate with AWS S3 using environment variables"""
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

            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )

            # Test connection
            self.s3_client.list_buckets()
            self.logger.info("✓ AWS S3 authentication successful")

        except NoCredentialsError:
            self.logger.error("Error: AWS credentials not found!")
            sys.exit(1)
        except ClientError as e:
            self.logger.error(f"Error: AWS authentication failed - {e}")
            sys.exit(1)

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

    def is_image_file(self, filename):
        """Check if file is an image based on extension"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
        return Path(filename).suffix.lower() in image_extensions

    def get_content_type(self, filename):
        """Get MIME type for file"""
        content_type, _ = mimetypes.guess_type(filename)
        return content_type or 'application/octet-stream'

    def upload_file_to_s3(self, file_path, bucket_name, s3_key):
        """Upload file to AWS S3 with proper content type"""
        try:
            file_size = os.path.getsize(file_path)
            content_type = self.get_content_type(file_path)

            with open(file_path, 'rb') as file:
                self.s3_client.upload_fileobj(
                    file,
                    bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': content_type}
                )

            return True, file_size

        except Exception as e:
            self.logger.error(f"Error uploading {file_path}: {e}")
            return False, 0

    def upload_property_images(self, property_id, s3_bucket=None, images_dir=None):
        """Upload all images for a specific property"""
        if s3_bucket is None:
            s3_bucket = self.bucket_name
        
        if images_dir is None:
            images_dir = os.getenv('IMAGE_FOLDER_NAME', 'images')

        property_folder = Path(images_dir) / property_id

        if not property_folder.exists():
            self.logger.warning(f"Property folder not found: {property_folder}")
            return False

        # Get all image files in the property folder
        image_files = [f for f in property_folder.iterdir() if f.is_file() and self.is_image_file(f.name)]

        if not image_files:
            self.logger.warning(f"No image files found in {property_folder}")
            return False

        self.logger.info(f"Found {len(image_files)} images for property {property_id}")

        uploaded_count = 0
        total_size = 0

        for image_file in image_files:
            # Create S3 key: property_id/image_name
            s3_key = f"{property_id}/{image_file.name}"

            success, file_size = self.upload_file_to_s3(image_file, s3_bucket, s3_key)

            if success:
                uploaded_count += 1
                total_size += file_size
                self.logger.info(f"✓ Uploaded {image_file.name} ({file_size} bytes)")
            else:
                self.logger.error(f"❌ Failed to upload {image_file.name}")

        self.logger.info(f"Property {property_id}: {uploaded_count}/{len(image_files)} images uploaded ({total_size} bytes total)")

        return uploaded_count > 0

    def upload_all_properties_from_seed(self, seed_data_path, s3_bucket=None, images_dir=None):
        """Upload images for all properties listed in seed.csv"""
        if s3_bucket is None:
            s3_bucket = self.bucket_name
        
        if images_dir is None:
            images_dir = os.getenv('IMAGE_FOLDER_NAME', 'images')

        self.logger.info("Starting upload for all properties from seed.csv...")

        # Load seed data
        parcel_mapping = self.load_seed_data(seed_data_path)
        
        if not parcel_mapping:
            self.logger.error("Failed to load seed data. Exiting.")
            return False

        images_path = Path(images_dir)

        # Check if images directory exists
        if not images_path.exists():
            self.logger.error(f"Error: Images directory '{images_dir}' not found")
            return False

        self.logger.info(f"✓ Found {len(parcel_mapping)} properties in seed.csv")

        # Upload each property
        total_successful = 0
        total_failed = 0

        for i, (property_id, address) in enumerate(parcel_mapping.items(), 1):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Processing Property {i}/{len(parcel_mapping)}: {property_id}")
            self.logger.info(f"Address: {address}")
            self.logger.info(f"{'=' * 60}")

            success = self.upload_property_images(property_id, s3_bucket, images_dir)

            if success:
                total_successful += 1
            else:
                total_failed += 1

        # Final summary
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"FINAL SUMMARY")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Total properties processed: {len(parcel_mapping)}")
        self.logger.info(f"Successful property uploads: {total_successful}")
        self.logger.info(f"Failed property uploads: {total_failed}")

        return total_successful > 0


def main():
    # Setup logging for main function
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/upload-to-s3.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Property Images Local to S3 Upload Script")
    logger.info("=" * 45)
    bucket_name = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai').rstrip('/')
    logger.info(f"Target S3 Bucket: {bucket_name}")
    logger.info("Mode: Upload all properties from seed.csv automatically")
    logger.info("=" * 45)

    # Check for required files
    seed_data_path = "seed.csv"
    
    if not os.path.exists(seed_data_path):
        logger.error(f"❌ {seed_data_path} not found!")
        logger.error("Please provide seed.csv with parcel_id,Address columns")
        sys.exit(1)
    
    logger.info(f"✓ Found {seed_data_path}")

    # Initialize uploader
    uploader = PropertyImagesUploader()

    # Authenticate with AWS
    logger.info("\n1. Authenticating with AWS S3...")
    uploader.authenticate_aws()

    # Configuration
    s3_bucket = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai')
    images_dir = os.getenv('IMAGE_FOLDER_NAME', 'images')

    logger.info(f"\n2. Configuration:")
    logger.info(f"   S3 Bucket: {s3_bucket}")
    logger.info(f"   Images Directory: {images_dir}")
    logger.info(f"   Seed Data: {seed_data_path}")
    logger.info(f"   S3 Prefix: [property_id]/")

    logger.info("\n3. Starting upload...")

    # Upload all properties from seed.csv
    success = uploader.upload_all_properties_from_seed(seed_data_path, s3_bucket, images_dir)

    if success:
        logger.info("\n🎉 Upload completed successfully!")
    else:
        logger.error("\n❌ Upload failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()