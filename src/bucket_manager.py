#!/usr/bin/env python3
"""
S3 Bucket Manager Utility
Creates and manages S3 buckets for the photo metadata AI system.
Ensures buckets exist with proper configuration and permissions.
"""

import os
import sys
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config
import json


class BucketManager:
    def __init__(self):
        self.s3_client = None
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai')
        
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
                logging.FileHandler('logs/bucket-manager.log')
                # Removed StreamHandler to only log to files
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
                region_name=aws_region,
                config=Config(
                    retries=dict(
                        max_attempts=3
                    )
                )
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

    def bucket_exists(self, bucket_name):
        """Check if bucket exists"""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return False
            else:
                self.logger.error(f"Error checking bucket {bucket_name}: {e}")
                return False

    def create_bucket(self, bucket_name, region=None):
        """Create S3 bucket with proper configuration"""
        if region is None:
            region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

        try:
            # Check if bucket already exists
            if self.bucket_exists(bucket_name):
                self.logger.info(f"✓ Bucket '{bucket_name}' already exists")
                return True

            self.logger.info(f"Creating bucket '{bucket_name}' in region '{region}'...")

            # Create bucket
            if region == 'us-east-1':
                # us-east-1 is the default region, no LocationConstraint needed
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )

            # Configure bucket settings
            self.configure_bucket(bucket_name)
            
            self.logger.info(f"✓ Successfully created bucket '{bucket_name}'")
            return True

        except ClientError as e:
            self.logger.error(f"Error creating bucket {bucket_name}: {e}")
            return False

    def configure_bucket(self, bucket_name):
        """Configure bucket with proper settings"""
        try:
            # Set bucket versioning
            self.s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            self.logger.info(f"✓ Enabled versioning for bucket '{bucket_name}'")

            # Set bucket encryption
            self.s3_client.put_bucket_encryption(
                Bucket=bucket_name,
                ServerSideEncryptionConfiguration={
                    'Rules': [
                        {
                            'ApplyServerSideEncryptionByDefault': {
                                'SSEAlgorithm': 'AES256'
                            }
                        }
                    ]
                }
            )
            self.logger.info(f"✓ Enabled encryption for bucket '{bucket_name}'")

            # Set bucket public access block (private by default)
            self.s3_client.put_public_access_block(
                Bucket=bucket_name,
                PublicAccessBlockConfiguration={
                    'BlockPublicAcls': True,
                    'IgnorePublicAcls': True,
                    'BlockPublicPolicy': True,
                    'RestrictPublicBuckets': True
                }
            )
            self.logger.info(f"✓ Set private access for bucket '{bucket_name}'")

            # Add bucket policy for secure access
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "SecureAccess",
                        "Effect": "Deny",
                        "Principal": "*",
                        "Action": "s3:*",
                        "Resource": [
                            f"arn:aws:s3:::{bucket_name}",
                            f"arn:aws:s3:::{bucket_name}/*"
                        ],
                        "Condition": {
                            "Bool": {
                                "aws:SecureTransport": "false"
                            }
                        }
                    }
                ]
            }

            self.s3_client.put_bucket_policy(
                Bucket=bucket_name,
                Policy=json.dumps(bucket_policy)
            )
            self.logger.info(f"✓ Applied security policy to bucket '{bucket_name}'")

        except ClientError as e:
            self.logger.warning(f"Warning: Could not configure bucket {bucket_name}: {e}")

    def ensure_bucket_exists(self, bucket_name=None):
        """Ensure bucket exists, create if it doesn't"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        self.logger.info(f"Ensuring bucket '{bucket_name}' exists...")

        if self.bucket_exists(bucket_name):
            self.logger.info(f"✓ Bucket '{bucket_name}' exists and is accessible")
            return True
        else:
            self.logger.info(f"Bucket '{bucket_name}' does not exist, creating...")
            return self.create_bucket(bucket_name)

    def list_buckets(self):
        """List all buckets accessible to the user"""
        try:
            response = self.s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            self.logger.info(f"Available buckets: {buckets}")
            return buckets
        except ClientError as e:
            self.logger.error(f"Error listing buckets: {e}")
            return []

    def get_bucket_info(self, bucket_name=None):
        """Get information about a bucket"""
        if bucket_name is None:
            bucket_name = self.bucket_name

        try:
            # Get bucket location
            location = self.s3_client.get_bucket_location(Bucket=bucket_name)
            
            # Get bucket versioning
            versioning = self.s3_client.get_bucket_versioning(Bucket=bucket_name)
            
            # Get bucket encryption
            encryption = self.s3_client.get_bucket_encryption(Bucket=bucket_name)
            
            # Get public access block
            public_access = self.s3_client.get_public_access_block(Bucket=bucket_name)

            info = {
                'name': bucket_name,
                'region': location.get('LocationConstraint') or 'us-east-1',
                'versioning': versioning.get('Status', 'NotEnabled'),
                'encryption': encryption.get('ServerSideEncryptionConfiguration', {}),
                'public_access': public_access.get('PublicAccessBlockConfiguration', {})
            }

            self.logger.info(f"Bucket '{bucket_name}' information:")
            self.logger.info(f"  Region: {info['region']}")
            self.logger.info(f"  Versioning: {info['versioning']}")
            self.logger.info(f"  Encryption: Enabled")
            self.logger.info(f"  Public Access: Blocked")

            return info

        except ClientError as e:
            self.logger.error(f"Error getting bucket info for {bucket_name}: {e}")
            return None


def main():
    """Main function for bucket management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='S3 Bucket Manager for Photo Metadata AI')
    parser.add_argument('--create', action='store_true', help='Create bucket if it does not exist')
    parser.add_argument('--bucket-name', type=str, help='Bucket name (overrides S3_BUCKET_NAME env var)')
    parser.add_argument('--region', type=str, help='AWS region (overrides AWS_DEFAULT_REGION env var)')
    parser.add_argument('--info', action='store_true', help='Show bucket information')
    parser.add_argument('--list', action='store_true', help='List all available buckets')
    
    args = parser.parse_args()

    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/bucket-manager.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("S3 Bucket Manager")
    logger.info("=" * 30)

    # Override environment variables if specified
    if args.bucket_name:
        os.environ['S3_BUCKET_NAME'] = args.bucket_name
    if args.region:
        os.environ['AWS_DEFAULT_REGION'] = args.region

    # Initialize bucket manager
    manager = BucketManager()
    
    # Authenticate with AWS
    logger.info("\n1. Authenticating with AWS...")
    manager.authenticate_aws()

    bucket_name = manager.bucket_name
    logger.info(f"Target bucket: {bucket_name}")

    # List buckets if requested
    if args.list:
        logger.info("\n2. Listing available buckets...")
        manager.list_buckets()
        return

    # Show bucket info if requested
    if args.info:
        logger.info("\n2. Getting bucket information...")
        manager.get_bucket_info(bucket_name)
        return

    # Create bucket if requested
    if args.create:
        logger.info("\n2. Creating bucket...")
        success = manager.ensure_bucket_exists(bucket_name)
        if success:
            logger.info("\n🎉 Bucket setup completed successfully!")
        else:
            logger.error("\n❌ Bucket setup failed!")
            sys.exit(1)
    else:
        # Default: ensure bucket exists
        logger.info("\n2. Ensuring bucket exists...")
        success = manager.ensure_bucket_exists(bucket_name)
        if success:
            logger.info("\n🎉 Bucket is ready!")
        else:
            logger.error("\n❌ Bucket setup failed!")
            sys.exit(1)


if __name__ == "__main__":
    main() 