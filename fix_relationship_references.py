#!/usr/bin/env python3
"""
Script to fix existing relationship files to use property.json as the reference.
This updates all relationship files in output directories to have the correct "from" field.
"""

import os
import json
import glob

def fix_relationship_file(filepath):
    """Fix a single relationship file to use property.json as the reference."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check if this is a relationship file that needs fixing
        if isinstance(data, dict) and "from" in data and "to" in data:
            # Check if the "from" field is incorrectly self-referencing
            from_ref = data.get("from", {}).get("/", "")
            to_ref = data.get("to", {}).get("/", "")
            
            # If "from" and "to" are the same (self-referencing), fix it
            if from_ref == to_ref and not from_ref.endswith("property.json"):
                # Update the "from" field to reference property.json
                data["from"] = {"/": "./property.json"}
                
                # Write the fixed data back to the file
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✅ Fixed: {filepath}")
                return True
            elif from_ref.endswith("property.json"):
                print(f"⏭️  Skipped (already correct): {filepath}")
                return False
            else:
                # Check if "from" should be property.json but isn't
                if not from_ref.endswith("property.json"):
                    data["from"] = {"/": "./property.json"}
                    
                    # Write the fixed data back to the file
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    print(f"✅ Fixed: {filepath}")
                    return True
                else:
                    print(f"⏭️  Skipped (already correct): {filepath}")
                    return False
        else:
            print(f"⏭️  Skipped (not a relationship file): {filepath}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

def fix_all_relationship_files():
    """Fix all relationship files in output directories."""
    # Find all relationship files in output directory
    output_dir = "output"
    total_fixed = 0
    total_processed = 0
    
    if not os.path.exists(output_dir):
        print(f"⚠️  Directory not found: {output_dir}")
        return
        
    print(f"\n🔧 Processing directory: {output_dir}")
    
    # Find all relationship files
    relationship_pattern = os.path.join(output_dir, "**", "relationship_*.json")
    relationship_files = glob.glob(relationship_pattern, recursive=True)
    
    if not relationship_files:
        print(f"   No relationship files found in {output_dir}")
        return
    
    print(f"   Found {len(relationship_files)} relationship files")
    
    for filepath in relationship_files:
        total_processed += 1
        if fix_relationship_file(filepath):
            total_fixed += 1
    
    print(f"\n📊 Summary:")
    print(f"   Total files processed: {total_processed}")
    print(f"   Total files fixed: {total_fixed}")
    print(f"   Total files skipped: {total_processed - total_fixed}")

if __name__ == "__main__":
    print("🔧 Fixing relationship file references in output directory...")
    fix_all_relationship_files()
    print("✅ Done!") 