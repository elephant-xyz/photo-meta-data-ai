#!/usr/bin/env python3
"""
Script to create property.json files in output directories that are missing them.
"""

import os
import json
import glob

def create_property_file(output_dir):
    """Create a property.json file in the given output directory."""
    property_path = os.path.join(output_dir, "property.json")
    
    # Check if property.json already exists
    if os.path.exists(property_path):
        print(f"⏭️  Property file already exists: {property_path}")
        return False
    
    # Extract property ID from directory name
    property_id = os.path.basename(output_dir)
    
    # Create property data
    property_data = {
        "parcel_id": property_id
    }
    
    # Write the property file
    with open(property_path, 'w') as f:
        json.dump(property_data, f, indent=2)
    
    print(f"✅ Created: {property_path}")
    return True

def create_all_property_files():
    """Create property.json files in all output directories."""
    output_dir = "output"
    total_created = 0
    total_processed = 0
    
    if not os.path.exists(output_dir):
        print(f"⚠️  Directory not found: {output_dir}")
        return
    
    print(f"\n🔧 Creating property.json files in: {output_dir}")
    
    # Find all subdirectories in output
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    if not subdirs:
        print(f"   No subdirectories found in {output_dir}")
        return
    
    print(f"   Found {len(subdirs)} subdirectories")
    
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        total_processed += 1
        if create_property_file(subdir_path):
            total_created += 1
    
    print(f"\n📊 Summary:")
    print(f"   Total directories processed: {total_processed}")
    print(f"   Total property files created: {total_created}")
    print(f"   Total directories skipped: {total_processed - total_created}")

if __name__ == "__main__":
    print("🔧 Creating property.json files in output directories...")
    create_all_property_files()
    print("✅ Done!") 