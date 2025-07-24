#!/usr/bin/env python3
"""
Unzip and Rename by Parcel ID Script

This script unzips submit.zip, extracts parcel IDs from property.json files,
and renames the folders to use parcel IDs instead of CIDs.
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
        logging.FileHandler('logs/unzip-count-data.log')
        # Removed StreamHandler to only log to files
    ]
)
logger = logging.getLogger(__name__)

def extract_parcel_id_from_property_json(property_file_path):
    """Extract parcel ID from a property.json file."""
    try:
        with open(property_file_path, 'r') as f:
            data = json.load(f)
        
        # Try different possible field names for parcel ID
        parcel_id = (
            data.get('parcel_identifier') or 
            data.get('request_identifier') or
            data.get('parcelId') or
            data.get('parcel_id') or
            data.get('id')
        )
        
        if parcel_id:
            logger.info(f"📋 Found parcel ID: {parcel_id} in {property_file_path}")
            return parcel_id
        else:
            logger.warning(f"⚠️  No parcel ID found in {property_file_path}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error reading {property_file_path}: {e}")
        return None

def unzip_and_rename_by_parcel_id():
    """Unzip submit.zip and rename folders to parcel IDs."""
    if not os.path.exists("submit.zip"):
        logger.error("❌ submit.zip not found")
        return False
    
    logger.info("📦 Unzipping submit.zip...")
    
    try:
        # Create extraction directory
        extract_dir = "county-data"
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        
        # Extract the zip file
        with zipfile.ZipFile("submit.zip", "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info("✅ Successfully unzipped submit.zip")
        
        # Find the submit directory inside the extracted content
        submit_dir = os.path.join(extract_dir, "submit")
        if not os.path.exists(submit_dir):
            logger.error(f"❌ submit directory not found in {extract_dir}")
            return False
        
        # Move all contents from submit/ to count-data/ directly
        logger.info(f"📁 Moving contents from submit/ to {extract_dir}/")
        for item in os.listdir(submit_dir):
            src = os.path.join(submit_dir, item)
            dst = os.path.join(extract_dir, item)
            shutil.move(src, dst)
        
        # Remove the empty submit directory
        os.rmdir(submit_dir)
        
        logger.info(f"📁 Processing property folders in {extract_dir}")
        
        # Process each CID folder
        renamed_count = 0
        for folder_name in os.listdir(extract_dir):
            folder_path = os.path.join(extract_dir, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
            
            logger.info(f"🔍 Processing folder: {folder_name}")
            
            # Look for property.json in this folder
            property_file = os.path.join(folder_path, "property.json")
            if not os.path.exists(property_file):
                logger.warning(f"⚠️  No property.json found in {folder_name}")
                continue
            
            # Extract parcel ID from property.json
            parcel_id = extract_parcel_id_from_property_json(property_file)
            if not parcel_id:
                logger.warning(f"⚠️  Could not extract parcel ID from {folder_name}")
                continue
            
            # Create new folder name using parcel ID (remove hyphens)
            new_folder_name = str(parcel_id).replace('-', '')
            new_folder_path = os.path.join(extract_dir, new_folder_name)
            
            # Check if target folder already exists
            if os.path.exists(new_folder_path):
                logger.warning(f"⚠️  Target folder already exists, removing: {new_folder_name}")
                shutil.rmtree(new_folder_path)
            
            # Rename the folder
            try:
                os.rename(folder_path, new_folder_path)
                logger.info(f"🔄 Renamed: {folder_name} -> {new_folder_name} (parcel ID: {parcel_id})")
                renamed_count += 1
            except Exception as e:
                logger.error(f"❌ Error renaming {folder_name}: {e}")
        
        logger.info(f"✅ Successfully renamed {renamed_count} folders")
        logger.info(f"📁 Final structure in {extract_dir}/")
        
        # List the final structure
        for item in os.listdir(extract_dir):
            item_path = os.path.join(extract_dir, item)
            if os.path.isdir(item_path):
                file_count = len([f for f in os.listdir(item_path) if f.endswith('.json')])
                logger.info(f"  📁 {item}/ ({file_count} JSON files)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing submit.zip: {e}")
        return False

def main():
    """Main function to orchestrate the unzip and rename process."""
    logger.info("🚀 Starting unzip and rename by parcel ID...")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Step 1: Unzip and rename folders
    logger.info("\n📦 Step 1: Unzipping and renaming folders...")
    if not unzip_and_rename_by_parcel_id():
        logger.error("❌ Failed to process submit.zip")
        return
    
    logger.info("\n✅ Complete unzip and rename process finished!")
    logger.info("📁 Files extracted to count-data/ with parcel ID folder names")

if __name__ == "__main__":
    main() 