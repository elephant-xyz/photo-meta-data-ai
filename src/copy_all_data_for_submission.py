#!/usr/bin/env python3
"""
Copy Property Files Script

This script handles the complete data preparation process:
1. Copy data from output to submit folder
2. Rename folders to propertyCID from upload results
3. Copy property.json files from submit.zip to the renamed folders
"""

import os
import json
import shutil
import csv
import zipfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/copy-all-data.log')
        # Removed StreamHandler to only log to files
    ]
)
logger = logging.getLogger(__name__)

def get_property_cid_from_upload_results():
    """Get property CIDs directly from upload results for folder naming."""
    upload_results_file = "upload-results.csv"
    if not os.path.exists(upload_results_file):
        logger.error(f"❌ Upload results file not found: {upload_results_file}")
        return {}
    
    property_cids = {}
    try:
        with open(upload_results_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = row.get('filePath', '')
                property_cid = row.get('propertyCid', '')
                
                if file_path and property_cid:
                    # Extract property ID from filePath like "/content/output/52434205310037080/..."
                    try:
                        parts = file_path.split('/')
                        for part in parts:
                            # Look for numeric property IDs (10+ digits)
                            if part.isdigit() and len(part) >= 10:
                                property_id = part
                                property_cids[property_id] = property_cid
                                logger.info(f"📋 Found property CID: {property_id} -> {property_cid}")
                                break
                    except Exception as e:
                        logger.warning(f"⚠️  Could not extract property ID from {file_path}: {e}")
    except Exception as e:
        logger.error(f"❌ Error reading upload results: {e}")
    
    return property_cids

def copy_data_to_submit_folder():
    """Copy data from output to submit folder."""
    output_dir = "output"
    submit_dir = "submit-photo"
    
    if not os.path.exists(output_dir):
        logger.error(f"❌ Output directory not found: {output_dir}")
        return False
    
    # Create submit directory
    os.makedirs(submit_dir, exist_ok=True)
    
    # Copy all contents from output to submit
    try:
        for item in os.listdir(output_dir):
            src = os.path.join(output_dir, item)
            dst = os.path.join(submit_dir, item)
            
            if os.path.isdir(src):
                if os.path.exists(dst):
                    # Don't remove the entire directory, just copy files
                    logger.info(f"📁 Directory {item} already exists, copying files...")
                    for root, dirs, files in os.walk(src):
                        # Calculate relative path
                        rel_path = os.path.relpath(root, src)
                        dst_root = os.path.join(dst, rel_path)
                        
                        # Create destination directories
                        for dir_name in dirs:
                            os.makedirs(os.path.join(dst_root, dir_name), exist_ok=True)
                        
                        # Copy files
                        for file_name in files:
                            src_file = os.path.join(root, file_name)
                            dst_file = os.path.join(dst_root, file_name)
                            shutil.copy2(src_file, dst_file)
                            logger.info(f"  📄 Copied: {file_name}")
                else:
                    shutil.copytree(src, dst)
                    logger.info(f"📁 Copied directory: {item}")
            else:
                shutil.copy2(src, dst)
                logger.info(f"📄 Copied file: {item}")
        
        logger.info(f"✅ Successfully copied data to {submit_dir}")
        return True
    except Exception as e:
        logger.error(f"❌ Error copying data: {e}")
        return False

def rename_folders_to_property_cid(property_cid_mapping):
    """Rename folders to propertyCID from upload results."""
    submit_dir = "submit-photo"
    
    if not os.path.exists(submit_dir):
        logger.error(f"❌ Submit directory not found: {submit_dir}")
        return False
    
    logger.info(f"📋 Available property CID mappings: {list(property_cid_mapping.keys())}")
    
    renamed_count = 0
    for folder_name in os.listdir(submit_dir):
        folder_path = os.path.join(submit_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        logger.info(f"🔍 Processing folder: {folder_name}")
        
        # Use folder name directly as parcel ID
        parcel_id = folder_name
        
        if parcel_id in property_cid_mapping:
            new_folder_name = property_cid_mapping[parcel_id]
            new_folder_path = os.path.join(submit_dir, new_folder_name)
            
            logger.info(f"📋 Mapping: {parcel_id} -> {new_folder_name}")
            logger.info(f"📁 Renaming: {folder_path} -> {new_folder_path}")
            
            if os.path.exists(new_folder_path):
                logger.warning(f"⚠️  Target folder already exists, removing: {new_folder_name}")
                shutil.rmtree(new_folder_path)
            
            if os.path.exists(folder_path):
                os.rename(folder_path, new_folder_path)
                logger.info(f"🔄 Renamed: {folder_name} -> {new_folder_name} (parcel ID: {parcel_id})")
                renamed_count += 1
            else:
                logger.error(f"❌ Source folder does not exist: {folder_path}")
        else:
            logger.warning(f"⚠️  No mapping found for folder: {folder_name} (parcel ID: {parcel_id})")
    
    logger.info(f"✅ Renamed {renamed_count} folders")
    return True

def copy_property_files_from_zip():
    """Unzip submit.zip, find property.json files, and copy to submit-photo property folders."""
    if not os.path.exists("submit.zip"):
        logger.warning("⚠️  submit.zip not found - skipping property.json copy")
        return
    
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
        
    except Exception as e:
        logger.error(f"❌ Error processing submit.zip: {e}")
        # Clean up on error
        if os.path.exists("submit-extracted"):
            shutil.rmtree("submit-extracted")

def main():
    """Main function to orchestrate the complete data preparation process."""
    logger.info("🚀 Starting complete data preparation process...")
    
    # Step 1: Load property CIDs from upload results
    logger.info("\n📋 Step 1: Loading property CIDs from upload results...")
    property_cids = get_property_cid_from_upload_results()
    if not property_cids:
        logger.error("❌ No property CIDs found. Please ensure upload-results.csv exists.")
        return
    
    # Step 2: Copy data from output to submit folder
    logger.info("\n📁 Step 2: Copying data from output to submit folder...")
    if not copy_data_to_submit_folder():
        logger.error("❌ Failed to copy data to submit folder")
        return
    
    # Step 3: Rename folders to propertyCID
    logger.info("\n🔄 Step 3: Renaming folders to propertyCID...")
    if not rename_folders_to_property_cid(property_cids):
        logger.error("❌ Failed to rename folders")
        return
    
    # Step 4: Copy property.json files from submit.zip to the renamed folders
    logger.info("\n📋 Step 4: Copying property.json files from submit.zip...")
    copy_property_files_from_zip()
    
    logger.info("\n✅ Complete data preparation process finished!")
    logger.info("📁 Submit folder: submit-photo")
    logger.info("🔍 Ready for schema validation and submission")

if __name__ == "__main__":
    main() 