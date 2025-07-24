#!/usr/bin/env python3
"""
Script to create missing property.json files in all output directories.
This will create property.json files with the parcel_id based on the directory name.
"""

import os
import json
import glob

def create_property_file_for_directory(directory):
    """Create property.json file for a specific directory."""
    property_id = os.path.basename(directory)
    property_path = os.path.join(directory, "property.json")
    
    # Check if property.json already exists
    if os.path.exists(property_path):
        print(f"  [✓] Already exists: {property_path}")
        return
    
    # Create property data
    property_data = {
        "parcel_id": property_id
    }
    
    try:
        with open(property_path, "w") as f:
            json.dump(property_data, f, indent=2)
        print(f"  [✓] Created: {property_path}")
    except Exception as e:
        print(f"  [!] Error creating {property_path}: {e}")

def create_missing_property_files():
    """Create missing property.json files in all output directories."""
    output_base = "output"
    
    if not os.path.exists(output_base):
        print(f"Output directory '{output_base}' not found!")
        return
    
    # Find all property directories
    property_dirs = []
    for item in os.listdir(output_base):
        item_path = os.path.join(output_base, item)
        if os.path.isdir(item_path):
            property_dirs.append(item_path)
    
    print(f"Found {len(property_dirs)} property directories to process")
    
    for property_dir in property_dirs:
        print(f"\nProcessing property: {os.path.basename(property_dir)}")
        create_property_file_for_directory(property_dir)

if __name__ == "__main__":
    print("🔧 Creating missing property.json files...")
    create_missing_property_files()
    print("\n✅ Finished creating property.json files!") 