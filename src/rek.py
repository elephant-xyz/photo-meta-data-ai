#!/usr/bin/env python3
"""
Script to analyze photos using AWS Rekognition and categorize them into folders
based on detected labels. Creates organized folder structure with labeled images.
"""

import os
import sys
import json
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from collections import defaultdict
import time


class RekognitionCategorizer:
    def __init__(self):
        self.s3_client = None
        self.rekognition_client = None
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai')

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
                print("Error: AWS credentials not found in environment variables!")
                print("Please set the following environment variables:")
                print("- AWS_ACCESS_KEY_ID")
                print("- AWS_SECRET_ACCESS_KEY")
                print("- AWS_DEFAULT_REGION (optional, defaults to us-east-1)")
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
            print("✓ AWS S3 authentication successful")
            print("✓ AWS Rekognition client initialized")
            print(f"✓ Using S3 bucket: {self.bucket_name}")

        except NoCredentialsError:
            print("Error: AWS credentials not found!")
            sys.exit(1)
        except ClientError as e:
            print(f"Error: AWS authentication failed - {e}")
            sys.exit(1)

    def get_property_images(self, property_id):
        """Get list of all images for a property from S3"""
        try:
            prefix = f"{property_id}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                print(f"No images found for property {property_id}")
                return []

            images = []
            for obj in response['Contents']:
                key = obj['Key']
                # Skip if it's just the folder itself
                if key.endswith('/'):
                    continue
                # Only process image files
                if self.is_image_file(key):
                    images.append(key)

            return images

        except ClientError as e:
            print(f"Error listing images for property {property_id}: {e}")
            return []

    def is_image_file(self, key):
        """Check if S3 key is an image file"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        return Path(key).suffix.lower() in image_extensions

    def analyze_image_with_rekognition(self, image_key):
        """Analyze image using AWS Rekognition"""
        try:
            response = self.rekognition_client.detect_labels(
                Image={
                    'S3Object': {
                        'Bucket': self.bucket_name,
                        'Name': image_key
                    }
                },
                MaxLabels=20,
                MinConfidence=70.0
            )

            labels = []
            for label in response['Labels']:
                labels.append({
                    'Name': label['Name'],
                    'Confidence': label['Confidence']
                })

            return labels

        except ClientError as e:
            print(f"Error analyzing image {image_key}: {e}")
            return []

    def categorize_image(self, labels):
        """Categorize image based on detected labels"""
        detected_labels = [label['Name'] for label in labels]

        # Check each category mapping
        for category, keywords in self.category_mappings.items():
            if category == 'other':
                continue

            # Check if any keyword matches detected labels
            for keyword in keywords:
                if keyword in detected_labels:
                    return category

        # If no category matched, return 'other'
        return 'other'

    def copy_image_to_category(self, source_key, property_id, category, image_name):
        """Copy image to categorized folder structure"""
        try:
            # Create destination key: property_id/category/image_name
            dest_key = f"{property_id}/{category}/{image_name}"

            # Copy the image
            copy_source = {
                'Bucket': self.bucket_name,
                'Key': source_key
            }

            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=dest_key
            )

            return True

        except ClientError as e:
            print(f"Error copying image {source_key} to {dest_key}: {e}")
            return False

    def process_property(self, property_id):
        """Process all images for a property"""
        print(f"\n{'=' * 60}")
        print(f"Processing Property: {property_id}")
        print(f"{'=' * 60}")

        # Get all images for the property
        images = self.get_property_images(property_id)

        if not images:
            print(f"No images found for property {property_id}")
            return False

        print(f"Found {len(images)} images to process")

        # Track categorization results
        categorization_results = defaultdict(list)
        processed_count = 0
        failed_count = 0

        for i, image_key in enumerate(images, 1):
            image_name = Path(image_key).name
            print(f"\n[{i}/{len(images)}] Processing: {image_name}")

            # Analyze image with Rekognition
            print("  Analyzing with Rekognition...")
            labels = self.analyze_image_with_rekognition(image_key)

            if not labels:
                print(f"  ❌ Failed to analyze {image_name}")
                failed_count += 1
                continue

            # Show top labels
            top_labels = sorted(labels, key=lambda x: x['Confidence'], reverse=True)[:5]
            label_strings = [f"{l['Name']} ({l['Confidence']:.1f}%)" for l in top_labels]
            print(f"  Top labels: {', '.join(label_strings)}")

            # Categorize image
            category = self.categorize_image(labels)
            print(f"  Category: {category}")

            # Copy to categorized folder
            print(f"  Copying to categorized folder...")
            success = self.copy_image_to_category(image_key, property_id, category, image_name)

            if success:
                print(f"  ✓ Successfully categorized as {category}")
                categorization_results[category].append({
                    'image': image_name,
                    'labels': labels
                })
                processed_count += 1
            else:
                print(f"  ❌ Failed to copy {image_name}")
                failed_count += 1

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        # Summary for this property
        print(f"\n{'=' * 50}")
        print(f"Property {property_id} Summary")
        print(f"{'=' * 50}")
        print(f"Total images: {len(images)}")
        print(f"Successfully processed: {processed_count}")
        print(f"Failed: {failed_count}")
        print(f"\nCategorization breakdown:")
        for category, items in categorization_results.items():
            print(f"  {category}: {len(items)} images")

        # Save categorization results to JSON
        self.save_categorization_results(property_id, categorization_results)

        return processed_count > 0

    def save_categorization_results(self, property_id, results):
        """Save categorization results to S3 as JSON"""
        try:
            results_data = {
                'property_id': property_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'categories': dict(results)
            }

            results_json = json.dumps(results_data, indent=2)
            results_key = f"{property_id}/categorization_results.json"

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=results_key,
                Body=results_json,
                ContentType='application/json'
            )

            print(f"✓ Categorization results saved to: s3://{self.bucket_name}/{results_key}")

        except ClientError as e:
            print(f"Error saving results: {e}")

    def get_all_properties(self):
        """Get list of all property IDs in the bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Delimiter='/'
            )

            properties = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    property_id = prefix['Prefix'].rstrip('/')
                    properties.append(property_id)

            return properties

        except ClientError as e:
            print(f"Error listing properties: {e}")
            return []


def main():
    print("AWS Rekognition Photo Categorizer")
    print("=" * 45)
    bucket_name = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai')
    print(f"Target S3 Bucket: {bucket_name}")
    print("=" * 45)

    # Initialize categorizer
    categorizer = RekognitionCategorizer()

    # Authenticate with AWS
    print("\n1. Authenticating with AWS...")
    categorizer.authenticate_aws()

    # Check if upload is needed
    print("\n2. Upload Options:")
    print("1. Upload images from local folder to S3")
    print("2. Process existing images in S3")
    print("3. Upload and then process")
    
    upload_choice = input("Choose option (1/2/3): ").strip()

    if upload_choice == "1":
        # Upload only
        from .uploadtoS3 import PropertyImagesUploader
        uploader = PropertyImagesUploader()
        uploader.authenticate_aws()
        success = uploader.upload_all_properties(bucket_name, "images")
        if success:
            print("\n🎉 Upload completed successfully!")
        else:
            print("\n❌ Upload failed!")
            sys.exit(1)
        return
    elif upload_choice == "3":
        # Upload and then process
        from .uploadtoS3 import PropertyImagesUploader
        uploader = PropertyImagesUploader()
        uploader.authenticate_aws()
        print("\n📤 Uploading images to S3...")
        upload_success = uploader.upload_all_properties(bucket_name, "images")
        if not upload_success:
            print("\n❌ Upload failed! Cannot proceed with categorization.")
            sys.exit(1)
        print("\n✅ Upload completed! Proceeding with categorization...")

    # Get user input for property ID
    print("\n3. Select property to process:")
    print("Enter property ID (or 'all' to process all properties):")

    property_input = input("Property ID: ").strip()

    if not property_input:
        print("Error: Property ID is required!")
        sys.exit(1)

    print("\n4. Starting image analysis and categorization...")

    success = False

    if property_input.lower() == 'all':
        # Process all properties
        properties = categorizer.get_all_properties()
        if not properties:
            print("No properties found in bucket")
            sys.exit(1)

        print(f"Found {len(properties)} properties to process")

        total_successful = 0
        total_failed = 0

        for i, property_id in enumerate(properties, 1):
            print(f"\n{'=' * 70}")
            print(f"Processing Property {i}/{len(properties)}: {property_id}")
            print(f"{'=' * 70}")

            if categorizer.process_property(property_id):
                total_successful += 1
            else:
                total_failed += 1

        print(f"\n{'=' * 70}")
        print(f"FINAL SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total properties processed: {len(properties)}")
        print(f"Successful: {total_successful}")
        print(f"Failed: {total_failed}")

        success = total_successful > 0
    else:
        # Process single property
        success = categorizer.process_property(property_input)

    if success:
        print("\n🎉 Image categorization completed successfully!")
        print("\nCategorized images are now organized in folders:")
        print("s3://photo-metadata-ai/PROPERTY_ID/CATEGORY/image.jpg")
    else:
        print("\n❌ Image categorization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()