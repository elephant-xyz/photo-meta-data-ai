#!/usr/bin/env python3
"""
Script to create image folders based on property IDs from upload_results.csv
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


def validate_upload_results_file(upload_results_path):
    """Validate that upload_results.csv exists and has required columns"""
    if not os.path.exists(upload_results_path):
        raise FileNotFoundError(f"Upload results file not found: {upload_results_path}")
    
    try:
        df = pd.read_csv(upload_results_path)
        print(f"✓ Loaded upload results file: {upload_results_path}")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['propertyCid', 'dataGroupCid', 'dataCid', 'filePath']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error reading upload results file: {e}")


def extract_property_ids_from_filepath(df):
    """Extract property IDs from filePath column"""
    property_ids = set()
    
    for filepath in df['filePath'].dropna():
        # Extract property ID from filepath like "/content/output/30434108090030050/..."
        try:
            # Split by '/' and look for the property ID pattern
            parts = filepath.split('/')
            for part in parts:
                # Look for numeric property IDs (10+ digits)
                if part.isdigit() and len(part) >= 10:
                    property_ids.add(part)
                    break
        except Exception as e:
            print(f"Warning: Could not extract property ID from {filepath}: {e}")
    
    return list(property_ids)


def create_folder_structure(property_ids, base_path, image_folder_name):
    """Create folder structure based on property IDs"""
    # Create the root image folder
    image_folder_path = os.path.join(base_path, image_folder_name)
    os.makedirs(image_folder_path, exist_ok=True)
    print(f"✓ Created root folder: {image_folder_path}")
    
    print(f"Found {len(property_ids)} unique property IDs")
    
    # Create subfolders for each property ID
    created_folders = []
    for property_id in property_ids:
        # Clean up property_id
        folder_name = str(property_id).strip()
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
    
    parser = argparse.ArgumentParser(description='Create image folders based on upload_results.csv')
    parser.add_argument('--upload-results-file', type=str, default='upload_results.csv', 
                       help='Path to upload_results.csv file (default: upload_results.csv)')
    parser.add_argument('--base-path', type=str, default='/content', 
                       help='Base path for image folders (default: /content)')
    parser.add_argument('--image-folder', type=str, default=None,
                       help='Image folder name (overrides IMAGE_FOLDER_NAME env var)')
    parser.add_argument('--env-file', type=str, default=None,
                       help='Path to .env file (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("Image Folder Setup Script (Updated for upload_results.csv)")
    logger.info("=" * 60)
    
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
    
    # Validate upload results file
    try:
        df = validate_upload_results_file(args.upload_results_file)
    except Exception as e:
        logger.error(f"❌ Upload results file validation failed: {e}")
        sys.exit(1)
    
    # Extract property IDs from filepath
    try:
        property_ids = extract_property_ids_from_filepath(df)
        if not property_ids:
            logger.error("❌ No property IDs found in upload_results.csv")
            sys.exit(1)
        
        logger.info(f"✓ Extracted {len(property_ids)} property IDs: {property_ids}")
        
    except Exception as e:
        logger.error(f"❌ Failed to extract property IDs: {e}")
        sys.exit(1)
    
    # Create folder structure
    try:
        created_folders = create_folder_structure(property_ids, args.base_path, image_folder_name)
        
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
        logger.info(f"   Place images in: {image_folder_path}/[property_id]/")
        
    except Exception as e:
        logger.error(f"❌ Folder creation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 