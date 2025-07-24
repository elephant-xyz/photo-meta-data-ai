#!/usr/bin/env python3
"""
Copy All Files from Submit Zip Script

This script unzips submit.zip and copies all JSON files to the corresponding 
property ID folders in submit-photo directory.
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/copy-from-zip.log')
        # Removed StreamHandler to only log to files
    ]
)
logger = logging.getLogger(__name__)

def unzip_and_copy_all_files():
    """Unzip submit.zip and copy all JSON files to submit-photo folders."""
    if not os.path.exists("submit.zip"):
        logger.error("❌ submit.zip not found")
        return False
    
    logger.info("📦 Unzipping submit.zip...")
    
    try:
        # Create temporary extraction directory
        extract_dir = "submit-extracted"
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        
        # Extract the zip file
        with zipfile.ZipFile("submit.zip", "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info("✅ Successfully unzipped submit.zip")
        
        # Find all JSON files in the extracted content
        json_files = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
        
        logger.info(f"📁 Found {len(json_files)} JSON files in submit.zip")
        
        # Copy JSON files to corresponding submit-photo folders
        copied_count = 0
        for json_file in json_files:
            # Extract the property CID from the path
            # Handle both possible path formats:
            # 1. submit-extracted/submit/bafkreixxx/filename.json
            # 2. submit-extracted/bafkreixxx/filename.json
            path_parts = json_file.split(os.sep)
            
            # Find the property CID (should be a folder name that looks like a CID)
            property_cid = None
            filename = path_parts[-1]  # The JSON filename
            
            # Look for a CID-like folder name (starts with 'bafkrei' and is in the right position)
            for i, part in enumerate(path_parts):
                if part.startswith('bafkrei') and i < len(path_parts) - 1:
                    property_cid = part
                    break
            
            if property_cid:
                # Target path in submit-photo (directly under submit-photo, no submit subfolder)
                target_dir = os.path.join("submit-photo", property_cid)
                target_file = os.path.join(target_dir, filename)
                
                # Create target directory if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)
                
                # Copy the JSON file
                try:
                    shutil.copy2(json_file, target_file)
                    logger.info(f"✅ Copied: {property_cid}/{filename}")
                    copied_count += 1
                except Exception as e:
                    logger.error(f"❌ Error copying {property_cid}/{filename}: {e}")
            else:
                logger.warning(f"⚠️  Could not extract property CID from path: {json_file}")
        
        logger.info(f"✅ Successfully copied {copied_count} JSON files to submit-photo folders")
        
        # Clean up extracted files
        shutil.rmtree(extract_dir)
        logger.info("🧹 Cleaned up extracted files")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing submit.zip: {e}")
        # Clean up on error
        if os.path.exists("submit-extracted"):
            shutil.rmtree("submit-extracted")
        return False

def main():
    """Main function to orchestrate the file copying process."""
    logger.info("🚀 Starting file copy from submit.zip...")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Step 1: Unzip and copy all files
    logger.info("\n📦 Step 1: Unzipping and copying all JSON files...")
    if not unzip_and_copy_all_files():
        logger.error("❌ Failed to copy files from submit.zip")
        return
    
    logger.info("\n✅ Complete file copy process finished!")
    logger.info("📁 All JSON files copied to submit-photo folders")

if __name__ == "__main__":
    main() 