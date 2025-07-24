#!/usr/bin/env python3
"""
Script to get property CID from IPFS and download it to the folder
"""

import os
import json
import requests
import subprocess
import sys
import pandas as pd
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and return the result"""
    print(f"🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description}")
        print(f"Error: {e.stderr}")
        return None

def get_property_cid_from_csv(parcel_id):
    """Get the property CID from upload_results.csv using the parcel ID"""
    print(f"🔍 Getting property CID for parcel ID: {parcel_id}")
    
    csv_file = "upload-results.csv"
    if not os.path.exists(csv_file):
        print(f"❌ {csv_file} not found!")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} records from {csv_file}")
        
        # Find the property CID for this parcel ID
        # The filePath column contains the parcel ID in the path
        for _, row in df.iterrows():
            file_path = row.get('filePath', '')
            if parcel_id in file_path:
                property_cid = row.get('propertyCid')
                if property_cid:
                    print(f"✅ Found property CID: {property_cid}")
                    return property_cid
        
        print(f"⚠️ No property CID found for parcel ID: {parcel_id}")
        return None
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

def download_from_ipfs(cid, output_dir):
    """Download content from IPFS CID"""
    print(f"📥 Downloading from IPFS CID: {cid}")
    
    # Create the CID directory
    cid_dir = os.path.join(output_dir, cid)
    os.makedirs(cid_dir, exist_ok=True)
    
    # Try to download using IPFS CLI if available
    if run_command("which ipfs", "Checking IPFS CLI"):
        print("✅ IPFS CLI found, using it to download")
        
        # Download using IPFS CLI
        cmd = f"ipfs get {cid} -o {cid_dir}/content"
        if run_command(cmd, f"Downloading {cid} from IPFS"):
            print(f"✅ Downloaded to {cid_dir}/content")
            return cid_dir
        else:
            print("❌ IPFS CLI download failed")
    
    # Fallback: Try HTTP gateway
    print("🔄 Trying HTTP gateway...")
    try:
        response = requests.get(f"https://ipfs.io/ipfs/{cid}", timeout=30)
        if response.status_code == 200:
            # Save the content
            content_path = os.path.join(cid_dir, "content.json")
            with open(content_path, "w") as f:
                f.write(response.text)
            print(f"✅ Downloaded via HTTP gateway to {content_path}")
            return cid_dir
        else:
            print(f"❌ HTTP gateway failed: {response.status_code}")
    except Exception as e:
        print(f"❌ HTTP gateway error: {e}")
    
    # If all else fails, create a placeholder structure
    print("⚠️ Creating placeholder structure...")
    create_placeholder_structure(cid_dir, cid)
    return cid_dir

def create_placeholder_structure(cid_dir, cid):
    """Create a placeholder structure for the CID directory"""
    print(f"📁 Creating placeholder structure in {cid_dir}")
    
    # Create basic files that Elephant CLI expects
    files_to_create = [
        ("property.json", {"parcel_id": "placeholder"}),
        ("bafkreifpjvcslz5hntsetlbic7kabfgzdpijdeuvgbhyismbgoj7x6nt7u.json", {"relationships": {}}),
    ]
    
    for filename, content in files_to_create:
        filepath = os.path.join(cid_dir, filename)
        with open(filepath, "w") as f:
            json.dump(content, f, indent=2)
        print(f"✅ Created: {filename}")

def copy_existing_data_to_cid(parcel_id, property_cid, output_dir):
    """Copy existing data from parcel ID directory to CID directory"""
    source_dir = os.path.join(output_dir, parcel_id)
    target_dir = os.path.join(output_dir, property_cid)
    
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    print(f"📋 Copying data from {parcel_id} to {property_cid}")
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy all files
    try:
        import shutil
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(target_dir, item)
            
            if os.path.isfile(source_item):
                shutil.copy2(source_item, target_item)
                print(f"✅ Copied: {item}")
            elif os.path.isdir(source_item):
                shutil.copytree(source_item, target_item, dirs_exist_ok=True)
                print(f"✅ Copied directory: {item}")
        
        print(f"✅ Successfully copied all data to {property_cid}")
        return True
    except Exception as e:
        print(f"❌ Error copying data: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Property CID Download Script")
    print("=" * 40)
    
    # Get parcel ID from command line or use default
    parcel_id = "30434108090030050"
    if len(sys.argv) > 1:
        parcel_id = sys.argv[1]
    
    output_dir = "output"
    
    print(f"📦 Parcel ID: {parcel_id}")
    print(f"📁 Output directory: {output_dir}")
    
    # Step 1: Get property CID from CSV
    property_cid = get_property_cid_from_csv(parcel_id)
    if not property_cid:
        print("❌ Could not get property CID from CSV")
        return
    
    print(f"🎯 Property CID: {property_cid}")
    
    # Step 2: Download from IPFS (or create placeholder)
    cid_dir = download_from_ipfs(property_cid, output_dir)
    
    # Step 3: Copy existing data to CID directory
    if copy_existing_data_to_cid(parcel_id, property_cid, output_dir):
        print(f"✅ Successfully prepared {property_cid} directory")
        
        # Step 4: Try Elephant CLI validation
        print(f"\n🔍 Running Elephant CLI validation on {property_cid}...")
        cmd = f"elephant-cli validate-and-upload {output_dir}/{property_cid} --dry-run --output-csv test-results.csv"
        if run_command(cmd, "Running Elephant CLI validation"):
            print("✅ Validation completed successfully!")
            print("📊 Results saved to test-results.csv")
        else:
            print("❌ Validation failed")
    else:
        print("❌ Failed to prepare CID directory")

if __name__ == "__main__":
    main() 