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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


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
                # Environment loaded successfully (no console output)
                env_loaded = True
                break
        
        if not env_loaded:
            # No .env file found (no console output)
            pass
        
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

    def process_property(self, property_id, address, max_workers=5):
        """Process all images for a property with multi-threading"""
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
        self.logger.info(f"🚀 Processing with {max_workers} workers for faster Rekognition...")
        
        # Process images with multi-threading
        categorization_results = {
            'property_id': property_id,
            'address': address,
            'total_images': len(images),
            'categorized_images': 0,
            'categories': defaultdict(int),
            'image_details': []
        }
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all image processing tasks
            future_to_image = {
                executor.submit(self.process_image_with_rekognition, image_key, property_id): image_key 
                for image_key in images
            }
            
            # Process completed tasks
            for future in as_completed(future_to_image):
                image_key = future_to_image[future]
                image_name = os.path.basename(image_key)
                
                try:
                    result = future.result()
                    
                    if result['success']:
                        categorization_results['categorized_images'] += 1
                        categorization_results['categories'][result['category']] += 1
                        categorization_results['image_details'].append(result['image_detail'])
                        
                        self.logger.info(f"✅ Categorized {image_name} as {result['category']}")
                    else:
                        self.logger.warning(f"⚠️  {result['error']}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error processing {image_name}: {e}")
        
        # Save categorization results
        self.save_categorization_results(property_id, categorization_results)
        
        self.logger.info(f"\n📊 Categorization Summary for {property_id}:")
        self.logger.info(f"   Total images: {categorization_results['total_images']}")
        self.logger.info(f"   Categorized: {categorization_results['categorized_images']}")
        self.logger.info(f"   Categories: {dict(categorization_results['categories'])}")
        
        return categorization_results

    def download_categorized_images_to_local(self, property_id):
        """Download categorized images from S3 back to local folders"""
        try:
            # Create local property folder
            local_property_path = os.path.join("images", property_id)
            os.makedirs(local_property_path, exist_ok=True)
            
            # List all categorized images in S3
            prefix = f"{property_id}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                self.logger.warning(f"No categorized images found for property {property_id}")
                return False
            
            downloaded_count = 0
            
            for obj in response['Contents']:
                s3_key = obj['Key']
                
                # Skip the original images and results files
                if s3_key.endswith('/') or s3_key.endswith('categorization_results.json'):
                    continue
                
                # Extract category from S3 key: property_id/category/image.jpg
                key_parts = s3_key.split('/')
                if len(key_parts) >= 3:
                    category = key_parts[1]
                    image_name = key_parts[2]
                    
                    # Create local category folder
                    local_category_path = os.path.join(local_property_path, category)
                    os.makedirs(local_category_path, exist_ok=True)
                    
                    # Download image to local
                    local_image_path = os.path.join(local_category_path, image_name)
                    
                    try:
                        self.s3_client.download_file(
                            self.bucket_name,
                            s3_key,
                            local_image_path
                        )
                        downloaded_count += 1
                        self.logger.info(f"✓ Downloaded {s3_key} to {local_image_path}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to download {s3_key}: {e}")
            
            self.logger.info(f"✅ Downloaded {downloaded_count} categorized images for property {property_id}")
            return downloaded_count > 0
            
        except Exception as e:
            self.logger.error(f"Error downloading categorized images for property {property_id}: {e}")
            return False

    def print_comprehensive_summary(self, all_results):
        """Print comprehensive summary of all properties and categories"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE CATEGORIZATION SUMMARY")
        print("="*80)
        
        total_properties = len(all_results)
        total_images = sum(result['total_images'] for result in all_results.values())
        total_categorized = sum(result['categorized_images'] for result in all_results.values())
        
        print(f"\n🏠 TOTAL PROPERTIES PROCESSED: {total_properties}")
        print(f"🖼️  TOTAL IMAGES: {total_images}")
        print(f"✅ TOTAL CATEGORIZED: {total_categorized}")
        print(f"📈 SUCCESS RATE: {(total_categorized/total_images*100):.1f}%" if total_images > 0 else "📈 SUCCESS RATE: N/A")
        
        # Overall category statistics
        all_categories = defaultdict(int)
        for result in all_results.values():
            for category, count in result['categories'].items():
                all_categories[category] += count
        
        print(f"\n📁 OVERALL CATEGORY BREAKDOWN:")
        for category, count in sorted(all_categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   {category}: {count} images")
        
        # Property-by-property breakdown
        print(f"\n🏠 PROPERTY-BY-PROPERTY BREAKDOWN:")
        print("-" * 80)
        
        for property_id, result in all_results.items():
            address = result.get('address', 'N/A')
            print(f"\n📍 Property: {property_id}")
            print(f"   Address: {address}")
            print(f"   Total Images: {result['total_images']}")
            print(f"   Categorized: {result['categorized_images']}")
            print(f"   Success Rate: {(result['categorized_images']/result['total_images']*100):.1f}%" if result['total_images'] > 0 else "   Success Rate: N/A")
            
            if result['categories']:
                print(f"   Categories:")
                for category, count in sorted(result['categories'].items(), key=lambda x: x[1], reverse=True):
                    print(f"     • {category}: {count} images")
            else:
                print(f"   Categories: None")
        
        print("\n" + "="*80)

    def process_image_with_rekognition(self, image_key, property_id):
        """Process a single image with Rekognition (for multi-threading)"""
        try:
            image_name = os.path.basename(image_key)
            
            # Analyze image with Rekognition
            labels = self.analyze_image_with_rekognition(image_key)
            
            if labels:
                # Categorize image
                category = self.categorize_image(labels)
                
                # Copy image to categorized folder
                if self.copy_image_to_category(image_key, property_id, category, image_name):
                    # Store image details
                    image_detail = {
                        'image_name': image_name,
                        'original_key': image_key,
                        'category': category,
                        'labels': [label['Name'] for label in labels],
                        'confidence_scores': [label['Confidence'] for label in labels]
                    }
                    
                    return {
                        'success': True,
                        'image_detail': image_detail,
                        'category': category
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Failed to copy {image_name} to category folder"
                    }
            else:
                return {
                    'success': False,
                    'error': f"No labels detected for {image_name}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Error processing {image_key}: {str(e)}"
            }

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

    def process_all_properties_from_seed(self, seed_data_path, max_workers=5):
        """Process all properties listed in seed.csv with comprehensive summary"""
        self.logger.info("Starting photo categorization workflow...")
        
        # Load seed data
        parcel_mapping = self.load_seed_data(seed_data_path)
        
        if not parcel_mapping:
            self.logger.error("Failed to load seed data. Exiting.")
            return False
        
        # Process each property and collect results
        all_results = {}
        total_properties = 0
        successful_properties = 0
        
        for parcel_id, address in parcel_mapping.items():
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Processing Property {total_properties + 1}: {parcel_id}")
            self.logger.info(f"Address: {address}")
            self.logger.info(f"{'='*80}")
            
            # Process the property with multi-threading
            result = self.process_property(parcel_id, address, max_workers)
            if result and result['categorized_images'] > 0:
                successful_properties += 1
                all_results[parcel_id] = result
                self.logger.info(f"✓ Successfully processed property {parcel_id}")
            else:
                self.logger.warning(f"⚠️  No images found or failed to process property {parcel_id}")
                # Add failed property to results for summary
                all_results[parcel_id] = {
                    'property_id': parcel_id,
                    'address': address,
                    'total_images': 0,
                    'categorized_images': 0,
                    'categories': defaultdict(int),
                    'image_details': []
                }
            
            total_properties += 1
        
        # Print comprehensive summary
        self.print_comprehensive_summary(all_results)
        
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
            logging.FileHandler('logs/photo-categorizer.log')
            # Removed StreamHandler to only log to files
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("AWS Rekognition Photo Categorizer")
    logger.info("=" * 45)
    bucket_name = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai').rstrip('/')
    logger.info(f"Target S3 Bucket: {bucket_name}")
    logger.info("=" * 45)

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

    # Get all properties from S3 bucket
    logger.info("\n3. Discovering properties in S3 bucket...")
    try:
        response = categorizer.s3_client.list_objects_v2(
            Bucket=bucket_name,
            Delimiter='/'
        )
        
        properties = []
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                property_id = prefix['Prefix'].rstrip('/')
                properties.append(property_id)
        
        if not properties:
            logger.error(f"❌ No properties found in S3 bucket {bucket_name}!")
            logger.error("Please ensure images are uploaded to S3 first.")
            sys.exit(1)
        
        logger.info(f"✓ Found {len(properties)} properties in S3 bucket")
        
    except Exception as e:
        logger.error(f"❌ Error discovering properties: {e}")
        sys.exit(1)

    # Process all properties found in S3
    logger.info("\n4. Processing all properties from S3...")
    logger.info("\n5. Starting image analysis and categorization with multi-threading...")
    
    all_results = {}
    total_properties = 0
    successful_properties = 0
    
    for property_id in properties:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing Property {total_properties + 1}: {property_id}")
        logger.info(f"{'='*80}")
        
        # Process the property with multi-threading
        result = categorizer.process_property(property_id, f"Property {property_id}", max_workers=5)
        if result and result['categorized_images'] > 0:
            successful_properties += 1
            all_results[property_id] = result
            logger.info(f"✓ Successfully processed property {property_id}")
        else:
            logger.warning(f"⚠️  No images found or failed to process property {property_id}")
            # Add failed property to results for summary
            all_results[property_id] = {
                'property_id': property_id,
                'address': f"Property {property_id}",
                'total_images': 0,
                'categorized_images': 0,
                'categories': defaultdict(int),
                'image_details': []
            }
        
        total_properties += 1
    
    # Print comprehensive summary
    categorizer.print_comprehensive_summary(all_results)
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info(f"FINAL SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total properties processed: {total_properties}")
    logger.info(f"Successful: {successful_properties}")
    logger.info(f"Failed: {total_properties - successful_properties}")
    
    if successful_properties > 0:
        logger.info("\n🎉 Image categorization completed successfully!")
        logger.info("\nCategorized images are now organized in S3:")
        logger.info(f"s3://{bucket_name}/property-id/category/image.jpg")
        
        # Download categorized images to local
        logger.info("\n6. Downloading categorized images to local folders...")
        
        for property_id in properties:
            logger.info(f"\n📥 Downloading categorized images for property {property_id}...")
            download_success = categorizer.download_categorized_images_to_local(property_id)
            if download_success:
                logger.info(f"✅ Successfully downloaded categorized images for {property_id}")
            else:
                logger.warning(f"⚠️  Failed to download categorized images for {property_id}")
        
        logger.info("\n🎉 All categorized images downloaded to local folders!")
        logger.info("Local structure: images/parcel_id/category/image.jpg")
    else:
        logger.error("\n❌ Image categorization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()