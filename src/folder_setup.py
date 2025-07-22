#!/usr/bin/env python3
"""
Script to create image folders based on parcel IDs from seed.csv
Can be run remotely via command line to set up the required folder structure.
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv


def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/folder-setup.log')
            # Removed StreamHandler to only log to files
        ]
    )
    return logging.getLogger(__name__)


def load_environment():
    """Load environment variables"""
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


def validate_seed_file(seed_path):
    """Validate that seed.csv exists and has required columns"""
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Seed file not found: {seed_path}")
    
    try:
        df = pd.read_csv(seed_path)
        print(f"✓ Loaded seed file: {seed_path}")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        
        if 'parcel_id' not in df.columns:
            raise ValueError("The seed.csv file must contain a 'parcel_id' column.")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error reading seed file: {e}")


def create_folder_structure(df, base_path, image_folder_name):
    """Create folder structure based on parcel IDs"""
    # Create the root image folder
    image_folder_path = os.path.join(base_path, image_folder_name)
    os.makedirs(image_folder_path, exist_ok=True)
    print(f"✓ Created root folder: {image_folder_path}")
    
    # Get unique parcel IDs
    parcel_ids = df['parcel_id'].dropna().unique()
    print(f"Found {len(parcel_ids)} unique parcel IDs")
    
    # Create subfolders for each parcel ID
    created_folders = []
    for parcel_id in parcel_ids:
        # Clean up parcel_id
        folder_name = str(parcel_id).strip()
        folder_path = os.path.join(image_folder_path, folder_name)
        
        try:
            os.makedirs(folder_path, exist_ok=True)
            created_folders.append(folder_name)
            print(f"✓ Created folder: {folder_name}")
        except Exception as e:
            print(f"❌ Failed to create folder {folder_name}: {e}")
    
    return created_folders


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create image folders based on seed.csv')
    parser.add_argument('--seed-file', type=str, default='seed.csv', 
                       help='Path to seed.csv file (default: seed.csv)')
    parser.add_argument('--base-path', type=str, default='/content', 
                       help='Base path for image folders (default: /content)')
    parser.add_argument('--image-folder', type=str, default=None,
                       help='Image folder name (overrides IMAGE_FOLDER_NAME env var)')
    parser.add_argument('--env-file', type=str, default=None,
                       help='Path to .env file (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("Image Folder Setup Script")
    logger.info("=" * 40)
    
    # Load environment variables
    if args.env_file:
        if os.path.exists(args.env_file):
            load_dotenv(args.env_file)
            logger.info(f"✓ Loaded environment from {args.env_file}")
        else:
            logger.error(f"❌ Environment file not found: {args.env_file}")
            sys.exit(1)
    else:
        load_environment()
    
    # Get image folder name
    image_folder_name = args.image_folder or os.getenv("IMAGE_FOLDER_NAME") or "images"
    logger.info(f"Image folder name: {image_folder_name}")
    
    # Validate seed file
    try:
        df = validate_seed_file(args.seed_file)
    except Exception as e:
        logger.error(f"❌ Seed file validation failed: {e}")
        sys.exit(1)
    
    # Create folder structure
    try:
        created_folders = create_folder_structure(df, args.base_path, image_folder_name)
        
        logger.info(f"\n{'='*50}")
        logger.info("FOLDER SETUP COMPLETED")
        logger.info(f"{'='*50}")
        logger.info(f"✅ Created {len(created_folders)} folders")
        logger.info(f"✅ Base path: {args.base_path}")
        logger.info(f"✅ Image folder: {image_folder_name}")
        logger.info(f"✅ Total path: {os.path.join(args.base_path, image_folder_name)}")
        
        # Show folder structure
        logger.info(f"\n📁 Folder structure:")
        image_folder_path = os.path.join(args.base_path, image_folder_name)
        for folder in sorted(created_folders):
            logger.info(f"  └── {folder}/")
        
        logger.info(f"\n🎉 Ready for image upload!")
        logger.info(f"   Place images in: {image_folder_path}/[parcel_id]/")
        
    except Exception as e:
        logger.error(f"❌ Folder creation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 