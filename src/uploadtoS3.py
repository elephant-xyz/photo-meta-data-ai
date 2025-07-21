#!/usr/bin/env python3
"""
Script to upload property images from local folder to AWS S3
Handles folder structure: images/PROPERTY_ID/<all images>
Automatically uploads all properties to photo-metadata-ai bucket
"""

import os
import sys
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import mimetypes


class PropertyImagesUploader:
    def __init__(self):
        self.s3_client = None
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai')

    def authenticate_aws(self):
        """Authenticate with AWS S3 using environment variables"""
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

            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )

            # Test connection
            self.s3_client.list_buckets()
            print("✓ AWS S3 authentication successful")

        except NoCredentialsError:
            print("Error: AWS credentials not found!")
            sys.exit(1)
        except ClientError as e:
            print(f"Error: AWS authentication failed - {e}")
            sys.exit(1)

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
            print(f"Error uploading {file_path}: {e}")
            return False, 0

    def upload_property_images(self, property_id, s3_bucket=None, images_dir="images"):
        if s3_bucket is None:
            s3_bucket = self.bucket_name
        """Upload all images for a specific property"""
        print(f"Starting upload for property ID: {property_id}")

        # Define the property folder path
        property_folder = Path(images_dir) / property_id

        # Check if property folder exists
        if not property_folder.exists():
            print(f"Error: Property folder '{property_folder}' not found")
            return False

        if not property_folder.is_dir():
            print(f"Error: '{property_folder}' is not a directory")
            return False

        print(f"✓ Found property folder: {property_folder}")

        # Get all image files in the property folder
        image_files = []
        for file_path in property_folder.iterdir():
            if file_path.is_file() and self.is_image_file(file_path.name):
                image_files.append(file_path)

        if not image_files:
            print("No image files found in property folder")
            return False

        print(f"✓ Found {len(image_files)} image files to upload")

        # Set up S3 prefix using folder name (property_id)
        s3_prefix = f"{property_id}/"

        # Upload each file
        successful_uploads = 0
        failed_uploads = 0
        total_size = 0

        for i, file_path in enumerate(image_files, 1):
            file_name = file_path.name
            file_size = file_path.stat().st_size

            print(f"\n[{i}/{len(image_files)}] Processing: {file_name}")
            print(f"Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

            # Upload to S3
            s3_key = f"{s3_prefix}{file_name}"
            print(f"  Uploading to S3: {s3_key}")

            success, uploaded_size = self.upload_file_to_s3(file_path, s3_bucket, s3_key)

            if success:
                print(f"  ✓ Successfully uploaded {file_name}")
                successful_uploads += 1
                total_size += uploaded_size
            else:
                print(f"  ❌ Failed to upload {file_name}")
                failed_uploads += 1

        # Summary
        print(f"\n{'=' * 50}")
        print(f"Upload Summary for Property {property_id}")
        print(f"{'=' * 50}")
        print(f"Total files found: {len(image_files)}")
        print(f"Successful uploads: {successful_uploads}")
        print(f"Failed uploads: {failed_uploads}")
        print(f"Total size uploaded: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
        print(f"S3 location: s3://{s3_bucket}/{s3_prefix}")

        return successful_uploads > 0

    def upload_all_properties(self, s3_bucket=None, images_dir="images"):
        if s3_bucket is None:
            s3_bucket = self.bucket_name
        """Upload images for all properties in the images directory"""
        print("Starting upload for all properties...")

        images_path = Path(images_dir)

        # Check if images directory exists
        if not images_path.exists():
            print(f"Error: Images directory '{images_dir}' not found")
            return False

        # Get all property folders
        property_folders = [d for d in images_path.iterdir() if d.is_dir()]

        if not property_folders:
            print("No property folders found in images directory")
            return False

        print(f"✓ Found {len(property_folders)} property folders")

        # Upload each property
        total_successful = 0
        total_failed = 0

        for i, property_folder in enumerate(property_folders, 1):
            property_id = property_folder.name
            print(f"\n{'=' * 60}")
            print(f"Processing Property {i}/{len(property_folders)}: {property_id}")
            print(f"{'=' * 60}")

            success = self.upload_property_images(property_id, s3_bucket, images_dir)

            if success:
                total_successful += 1
            else:
                total_failed += 1

        # Final summary
        print(f"\n{'=' * 60}")
        print(f"FINAL SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total properties processed: {len(property_folders)}")
        print(f"Successful property uploads: {total_successful}")
        print(f"Failed property uploads: {total_failed}")

        return total_successful > 0


def main():
    print("Property Images Local to S3 Upload Script")
    print("=" * 45)
    bucket_name = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai')
    print(f"Target S3 Bucket: {bucket_name}")
    print("Mode: Upload all properties automatically")
    print("=" * 45)

    # Initialize uploader
    uploader = PropertyImagesUploader()

    # Authenticate with AWS
    print("\n1. Authenticating with AWS S3...")
    uploader.authenticate_aws()

    # Configuration
    s3_bucket = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai')
    images_dir = "images"

    print(f"\n2. Configuration:")
    print(f"   S3 Bucket: {s3_bucket}")
    print(f"   Images Directory: {images_dir}")
    print(f"   S3 Prefix: [property_id]/")

    print("\n3. Starting upload...")

    # Upload all properties
    success = uploader.upload_all_properties(s3_bucket, images_dir)

    if success:
        print("\n🎉 Upload completed successfully!")
    else:
        print("\n❌ Upload failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()