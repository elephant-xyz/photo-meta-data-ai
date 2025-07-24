#!/usr/bin/env python3
"""
Copy Property Files from Submit.zip Script

This script unzips submit.zip and copies property.json files to the corresponding 
folders in submit-photo based on property IDs.
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
        logging.FileHandler('logs/copy-property-files.log')
        # Removed StreamHandler to only log to files
    ]
)
logger = logging.getLogger(__name__)

def extract_parcel_id_from_property_json(property_json_path):
    """Extract parcel ID from property.json file."""
    try:
        with open(property_json_path, 'r') as f:
            data = json.load(f)
        
        # Try different possible field names for parcel ID
        parcel_id = (
            data.get('parcel_identifier') or 
            data.get('request_identifier') or
            data.get('parcelId') or
            data.get('parcel_id')
        )
        
        if parcel_id:
            logger.info(f"📋 Found parcel ID: {parcel_id} in {property_json_path}")
            return parcel_id
        else:
            logger.warning(f"⚠️  No parcel ID found in {property_json_path}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error reading {property_json_path}: {e}")
        return None

def copy_property_files_from_zip():
    """Unzip submit.zip and copy property.json files to submit-photo folders."""
    
    # Check if submit.zip exists
    if not os.path.exists("submit.zip"):
        logger.error("❌ submit.zip not found")
        return False
    
    # Check if submit-photo directory exists
    if not os.path.exists("submit-photo"):
        logger.error("❌ submit-photo directory not found")
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
        
        # Find all property.json files in the extracted content
        property_files = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file == "property.json":
                    property_files.append(os.path.join(root, file))
        
        logger.info(f"📁 Found {len(property_files)} property.json files in submit.zip")
        
        # Copy property.json files to corresponding submit-photo folders
        copied_count = 0
        for property_file in property_files:
            # Extract the property CID from the path
            # Handle both possible path formats:
            # 1. submit-extracted/submit/bafkreixxx/property.json
            # 2. submit-extracted/bafkreixxx/property.json
            path_parts = property_file.split(os.sep)
            
            # Find the property CID (should be a folder name that looks like a CID)
            property_cid = None
            
            # Look for a CID-like folder name (starts with 'bafkrei' and is in the right position)
            for i, part in enumerate(path_parts):
                if part.startswith('bafkrei') and i < len(path_parts) - 1:
                    property_cid = part
                    break
            
            if property_cid:
                # Target path in submit-photo (directly under submit-photo, no submit subfolder)
                target_dir = os.path.join("submit-photo", property_cid)
                target_file = os.path.join(target_dir, "property.json")
                
                # Create target directory if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)
                
                # Copy the property.json file
                try:
                    shutil.copy2(property_file, target_file)
                    logger.info(f"✅ Copied: {property_cid}/property.json")
                    copied_count += 1
                except Exception as e:
                    logger.error(f"❌ Error copying {property_cid}/property.json: {e}")
            else:
                logger.warning(f"⚠️  Could not extract property CID from path: {property_file}")
        
        logger.info(f"✅ Successfully copied {copied_count} property.json files to submit-photo folders")
        
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
    """Main function to orchestrate the property file copying process."""
    logger.info("🚀 Starting property file copying from submit.zip...")
    
    # Step 1: Copy property.json files from submit.zip
    logger.info("\n📋 Step 1: Copying property.json files from submit.zip...")
    if not copy_property_files_from_zip():
        logger.error("❌ Failed to copy property files from submit.zip")
        return
    
    logger.info("\n✅ Property file copying process finished!")
    logger.info("📁 Property files copied to submit-photo folders")

if __name__ == "__main__":
    main() 