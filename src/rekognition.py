#!/usr/bin/env python3
"""
Script to analyze photos using AWS Rekognition and categorize them into folders
based on detected labels. Creates organized folder structure with labeled images.
"""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from collections import defaultdict
import time
from dotenv import load_dotenv


class RekognitionCategorizer:
    def __init__(self):
        self.s3_client = None
        self.rekognition_client = None
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
                logging.FileHandler(f'logs/photo-categorizer.log')
                # Removed StreamHandler to only log to files
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Define category mappings for common real estate photo types
        self.category_mappings = {
            'kitchen': ['Kitchen', 'Refrigerator', 'Oven', 'Microwave', 'Dishwasher', 'Sink', 'Cabinet', 'Countertop'],
            'bedroom': ['Bedroom', 'Bed', 'Mattress', 'Pillow', 'Blanket', 'Nightstand', 'Dresser'],
            'bathroom': ['Bathroom', 'Toilet', 'Bathtub', 'Shower', 'Sink', 'Mirror', 'Towel'],
            'living_room': ['Living Room', 'Sofa', 'Couch', 'Chair', 'Coffee Table', 'Television', 'Fireplace'],
            'dining_room': ['Dining Room', 'Dining Table', 'Chair', 'Chandelier'],
            'exterior': ['Building', 'House', 'Architecture', 'Roof', 'Window', 'Door', 'Garden', 'Lawn', 'Tree'],
            'garage': ['Garage', 'Car', 'Vehicle', 'Automobile'],
            'office': ['Office', 'Desk', 'Computer', 'Monitor', 'Chair'],
            'laundry': ['Laundry', 'Washing Machine', 'Dryer'],
            'stairs': ['Staircase', 'Stairs', 'Banister', 'Railing'],
            'closet': ['Closet', 'Wardrobe', 'Clothing', 'Hanger'],
            'pool': ['Pool', 'Swimming Pool', 'Water'],
            'balcony': ['Balcony', 'Terrace', 'Patio', 'Deck'],
            'other': []  # Default category for unmatched items
        }

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

            # Initialize Rekognition client
            self.rekognition_client = boto3.client(
                'rekognition',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )

            # Test connections
            self.s3_client.list_buckets()
            self.logger.info("✓ AWS S3 authentication successful")
            self.logger.info("✓ AWS Rekognition client initialized")
            self.logger.info(f"✓ Using S3 bucket: {self.bucket_name}")

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

    def get_property_images(self, property_id):
        """Get list of all images for a property from S3"""
        try:
            prefix = f"{property_id}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                self.logger.warning(f"No images found for property {property_id}")
                return []

            images = []
            for obj in response['Contents']:
                key = obj['Key']
                if self.is_image_file(key):
                    images.append(key)

            self.logger.info(f"Found {len(images)} images for property {property_id}")
            return images

        except ClientError as e:
            self.logger.error(f"Error listing images for property {property_id}: {e}")
            return []

    def is_image_file(self, key):
        """Check if file is an image based on extension"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        return Path(key).suffix.lower() in image_extensions

    def analyze_image_with_rekognition(self, image_key):
        """Analyze image using AWS Rekognition"""
        try:
            response = self.rekognition_client.detect_labels(
                Image={'S3Object': {'Bucket': self.bucket_name, 'Name': image_key}},
                MaxLabels=10,
                MinConfidence=70
            )
            return response['Labels']
        except ClientError as e:
            self.logger.error(f"Error analyzing image {image_key}: {e}")
            return []

    def categorize_image(self, labels):
        """Categorize image based on detected labels"""
        label_names = [label['Name'] for label in labels]
        
        for category, keywords in self.category_mappings.items():
            for keyword in keywords:
                if keyword.lower() in [name.lower() for name in label_names]:
                    return category
        
        return 'other'

    def copy_image_to_category(self, source_key, property_id, category, image_name):
        """Copy image to categorized folder in S3"""
        try:
            destination_key = f"{property_id}/{category}/{image_name}"
            
            self.s3_client.copy_object(
                Bucket=self.bucket_name,
                CopySource={'Bucket': self.bucket_name, 'Key': source_key},
                Key=destination_key
            )
            
            self.logger.info(f"✓ Copied {source_key} to {destination_key}")
            return True
            
        except ClientError as e:
            self.logger.error(f"Error copying image {source_key}: {e}")
            return False

    def process_property(self, property_id, address):
        """Process all images for a property"""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Processing Property: {property_id}")
        self.logger.info(f"Address: {address}")
        self.logger.info(f"{'=' * 60}")

        # Get all images for this property
        images = self.get_property_images(property_id)
        
        if not images:
            self.logger.warning(f"⚠️  No images found for property {property_id}, skipping...")
            return False
        
        self.logger.info(f"📁 Found {len(images)} images for property {property_id}")
        
        # Process each image
        categorization_results = {
            'property_id': property_id,
            'address': address,
            'total_images': len(images),
            'categorized_images': 0,
            'categories': defaultdict(int),
            'image_details': []
        }
        
        for image_key in images:
            image_name = os.path.basename(image_key)
            self.logger.info(f"\n🖼️  Processing image: {image_name}")
            
            # Analyze image with Rekognition
            labels = self.analyze_image_with_rekognition(image_key)
            
            if labels:
                # Categorize image
                category = self.categorize_image(labels)
                
                # Copy image to categorized folder
                if self.copy_image_to_category(image_key, property_id, category, image_name):
                    categorization_results['categorized_images'] += 1
                    categorization_results['categories'][category] += 1
                    
                    # Store image details
                    image_detail = {
                        'image_name': image_name,
                        'original_key': image_key,
                        'category': category,
                        'labels': [label['Name'] for label in labels],
                        'confidence_scores': [label['Confidence'] for label in labels]
                    }
                    categorization_results['image_details'].append(image_detail)
                    
                    self.logger.info(f"✅ Categorized {image_name} as {category}")
                else:
                    self.logger.error(f"❌ Failed to copy {image_name} to category folder")
            else:
                self.logger.warning(f"⚠️  No labels detected for {image_name}")
        
        # Save categorization results
        self.save_categorization_results(property_id, categorization_results)
        
        self.logger.info(f"\n📊 Categorization Summary for {property_id}:")
        self.logger.info(f"   Total images: {categorization_results['total_images']}")
        self.logger.info(f"   Categorized: {categorization_results['categorized_images']}")
        self.logger.info(f"   Categories: {dict(categorization_results['categories'])}")
        
        return categorization_results['categorized_images'] > 0

    def save_categorization_results(self, property_id, results):
        """Save categorization results as JSON in S3"""
        try:
            results_key = f"{property_id}/categorization_results.json"
            
            # Convert defaultdict to regular dict for JSON serialization
            results['categories'] = dict(results['categories'])
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=results_key,
                Body=json.dumps(results, indent=2),
                ContentType='application/json'
            )
            
            self.logger.info(f"✓ Saved categorization results to {results_key}")
            
        except ClientError as e:
            self.logger.error(f"Error saving categorization results for {property_id}: {e}")

    def process_all_properties_from_seed(self, seed_data_path):
        """Process all properties listed in seed.csv"""
        self.logger.info("Starting photo categorization workflow...")
        
        # Load seed data
        parcel_mapping = self.load_seed_data(seed_data_path)
        
        if not parcel_mapping:
            self.logger.error("Failed to load seed data. Exiting.")
            return False
        
        # Process each property
        total_properties = 0
        successful_properties = 0
        
        for parcel_id, address in parcel_mapping.items():
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Processing Property {total_properties + 1}: {parcel_id}")
            self.logger.info(f"Address: {address}")
            self.logger.info(f"{'='*80}")
            
            # Process the property
            if self.process_property(parcel_id, address):
                successful_properties += 1
                self.logger.info(f"✓ Successfully processed property {parcel_id}")
            else:
                self.logger.warning(f"⚠️  No images found or failed to process property {parcel_id}")
            
            total_properties += 1
        
        # Final summary
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"FINAL SUMMARY")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Total properties processed: {total_properties}")
        self.logger.info(f"Successful: {successful_properties}")
        self.logger.info(f"Failed: {total_properties - successful_properties}")
        
        return successful_properties > 0


def main():
    # Setup logging for main function
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/photo-categorizer.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("AWS Rekognition Photo Categorizer")
    logger.info("=" * 45)
    bucket_name = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai').rstrip('/')
    logger.info(f"Target S3 Bucket: {bucket_name}")
    logger.info("=" * 45)

    # Check for required files
    seed_data_path = "seed.csv"
    
    if not os.path.exists(seed_data_path):
        logger.error(f"❌ {seed_data_path} not found!")
        logger.error("Please provide seed.csv with parcel_id,Address columns")
        sys.exit(1)
    
    logger.info(f"✓ Found {seed_data_path}")

    # Initialize categorizer
    categorizer = RekognitionCategorizer()

    # Authenticate with AWS
    logger.info("\n1. Authenticating with AWS...")
    categorizer.authenticate_aws()

    # Ensure bucket exists
    logger.info("\n2. Ensuring S3 bucket exists...")
    from .bucket_manager import BucketManager
    bucket_manager = BucketManager()
    bucket_manager.authenticate_aws()
    bucket_success = bucket_manager.ensure_bucket_exists(bucket_name)
    if not bucket_success:
        logger.error("\n❌ Bucket setup failed! Cannot proceed.")
        sys.exit(1)
    logger.info("\n✅ Bucket is ready!")

    # Auto-upload and process all properties
    logger.info("\n3. Auto-processing all properties from seed.csv...")
    
    # Upload all properties first
    from .uploadtoS3 import PropertyImagesUploader
    uploader = PropertyImagesUploader()
    uploader.authenticate_aws()
    logger.info("\n📤 Uploading all images to S3...")
    upload_success = uploader.upload_all_properties_from_seed(seed_data_path, bucket_name, "images")
    if not upload_success:
        logger.error("\n❌ Upload failed! Cannot proceed with categorization.")
        sys.exit(1)
    logger.info("\n✅ Upload completed! Proceeding with categorization...")

    # Process all properties from seed.csv
    logger.info("\n4. Processing all properties from seed.csv...")
    logger.info("\n5. Starting image analysis and categorization...")
    
    success = categorizer.process_all_properties_from_seed(seed_data_path)

    if success:
        logger.info("\n🎉 Image categorization completed successfully!")
        logger.info("\nCategorized images are now organized in folders:")
        logger.info(f"s3://{bucket_name}/property-id/category/image.jpg")
    else:
        logger.error("\n❌ Image categorization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()