#!/usr/bin/env python3
"""
Simple Colab folder setup script
Run with: !folder-setup
"""

import os
import json
import requests
import pandas as pd
import argparse


def fetch_from_ipfs(cid):
    """Fetch content from IPFS using the CID"""
    try:
        url = f"https://ipfs.io/ipfs/{cid}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching from IPFS CID {cid}: {e}")
        return None


def follow_ipfs_chain(property_cid):
    """Follow the IPFS chain to get the final property JSON"""
    try:
        # Step 1: Get propertyCid from IPFS
        print(f"  [1] Fetching propertyCid: {property_cid}")
        property_data = fetch_from_ipfs(property_cid)
        if not property_data:
            return None, None

        # Step 2: Find property_seed value
        property_seed = None

        # Check in relationships (this is where it is in the actual data)
        if "relationships" in property_data and "property_seed" in property_data["relationships"]:
            prop_seed_obj = property_data["relationships"]["property_seed"]
            if isinstance(prop_seed_obj, dict) and "/" in prop_seed_obj:
                property_seed = prop_seed_obj["/"]
            elif isinstance(prop_seed_obj, str):
                property_seed = prop_seed_obj
            else:
                print(f"  [!] Unexpected property_seed format: {prop_seed_obj}")
                return None, None
        # Check direct property_seed field (fallback)
        elif "property_seed" in property_data:
            property_seed = property_data["property_seed"]

        if not property_seed:
            print(f"  [!] No property_seed found in {property_cid}")
            print(f"  [!] Available fields: {list(property_data.keys())}")
            return None, None

        print(f"  [2] Found property_seed: {property_seed}")

        # Step 3: Search IPFS with property_seed CID
        print(f"  [3] Fetching property_seed CID: {property_seed}")
        seed_data = fetch_from_ipfs(property_seed)
        if not seed_data:
            return None, None

        # Step 4: Get the 'from' field
        from_field = seed_data.get("from")
        if not from_field:
            print(f"  [!] No 'from' field found in {property_seed}")
            return None, None

        # Handle different formats of 'from' field
        if isinstance(from_field, dict) and "/" in from_field:
            from_cid = from_field["/"]
        elif isinstance(from_field, dict):
            from_cid = from_field.get("path", "").replace("./", "").replace(".json", "")
        elif isinstance(from_field, str):
            from_cid = from_field.replace("./", "").replace(".json", "")
        else:
            print(f"  [!] Unexpected 'from' field format: {from_field}")
            return None, None

        print(f"  [4] Found 'from' CID: {from_cid}")

        # Step 5: Search IPFS with that CID and get JSON content
        print(f"  [5] Fetching final JSON from: {from_cid}")
        final_json = fetch_from_ipfs(from_cid)
        if not final_json:
            return None, None

        print("  [✓] Successfully retrieved JSON content")
        return from_cid, final_json

    except Exception as e:
        print(f"  [!] Error following IPFS chain: {e}")
        return None, None


def extract_property_data(df):
    """Extract property IDs and CIDs from upload_results.csv"""
    property_data = {}  # {property_id: property_cid}

    for _, row in df.iterrows():
        filepath = row.get("filePath", "")
        property_cid = row.get("propertyCid", "")

        if filepath and property_cid:
            # Extract property ID from filepath like "/content/output/30434108090030050/..."
            try:
                parts = filepath.split("/")
                for part in parts:
                    # Look for numeric property IDs (10+ digits)
                    if part.isdigit() and len(part) >= 10:
                        property_id = part
                        property_data[property_id] = property_cid
                        break
            except Exception as e:
                print(f"Warning: Could not extract property ID from {filepath}: {e}")

    return property_data


def create_folders_and_cid_files(property_data, base_path="."):
    """Create folder structure and property CID files"""
    # Create the root image folder for images
    image_folder_path = os.path.join(base_path, "images")
    os.makedirs(image_folder_path, exist_ok=True)
    print(f"✓ Created images folder: {image_folder_path}")

    # Create the output folder for CID files
    output_folder_path = os.path.join(base_path, "output")
    os.makedirs(output_folder_path, exist_ok=True)
    print(f"✓ Created output folder: {output_folder_path}")

    print(f"Found {len(property_data)} unique properties")

    # Create subfolders and CID files for each property
    created_image_folders = []
    created_cid_files = []

    for property_id, property_cid in property_data.items():
        # Clean up property_id
        folder_name = str(property_id).strip()

        # Create image folder
        image_folder_path_property = os.path.join(image_folder_path, folder_name)
        os.makedirs(image_folder_path_property, exist_ok=True)
        created_image_folders.append(folder_name)
        print(f"✓ Created image folder: {folder_name}")

        # Create output folder for this property
        output_folder_path_property = os.path.join(output_folder_path, folder_name)
        os.makedirs(output_folder_path_property, exist_ok=True)
        print(f"✓ Created output folder: {folder_name}")

        try:
            # Follow IPFS chain to get the final JSON content
            print(f"\n🔍 Following IPFS chain for property {property_id}:")
            final_cid, final_json = follow_ipfs_chain(property_cid)

            if final_cid and final_json:
                # Create property CID file with the final CID in output folder
                cid_filename = f"{final_cid}.json"
                cid_file_path = os.path.join(output_folder_path_property, cid_filename)

                with open(cid_file_path, "w") as f:
                    json.dump(final_json, f, indent=2)

                created_cid_files.append(cid_filename)
                print(f"✓ Created CID file: {cid_filename} in output/{folder_name}/")
            else:
                print(f"⚠️  Could not retrieve JSON for property {property_id}")

        except Exception as e:
            print(f"❌ Failed to create CID file for {folder_name}: {e}")

    return created_image_folders, created_cid_files


def main():
    """Main function for Colab"""
    parser = argparse.ArgumentParser(description="Colab Folder Setup Script")
    parser.add_argument(
        "--base-path", type=str, default=".", help="Base path for creating folders (default: current directory)"
    )
    parser.add_argument(
        "--upload-results-file",
        type=str,
        default="upload-results.csv",
        help="Path to upload_results.csv file (default: upload-results.csv)",
    )

    args = parser.parse_args()

    print("🚀 Starting Colab Folder Setup...")
    print("=" * 50)

    # Check if upload_results.csv exists
    upload_results_path = os.path.join(args.base_path, args.upload_results_file)
    if not os.path.exists(upload_results_path):
        print(f"❌ {args.upload_results_file} not found!")
        print(f"Please create {args.upload_results_file} with your property data first.")
        return

    # Load upload results
    try:
        df = pd.read_csv(upload_results_path)
        print(f"✓ Loaded upload results file: {args.upload_results_file}")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading {args.upload_results_file}: {e}")
        return

    # Extract property data
    try:
        property_data = extract_property_data(df)
        if not property_data:
            print(f"❌ No property data found in {args.upload_results_file}")
            return

        property_ids = list(property_data.keys())
        property_cids = list(property_data.values())
        print(f"✓ Extracted {len(property_data)} properties:")
        for prop_id, prop_cid in property_data.items():
            print(f"  - Property ID: {prop_id} -> CID: {prop_cid}")

    except Exception as e:
        print(f"❌ Failed to extract property data: {e}")
        return

    # Create folder structure and CID files
    try:
        created_image_folders, created_cid_files = create_folders_and_cid_files(property_data, args.base_path)

        print(f"\n{'=' * 50}")
        print("FOLDER SETUP COMPLETED")
        print(f"{'=' * 50}")
        print(f"✅ Created {len(created_image_folders)} image folders")
        print(f"✅ Created {len(created_cid_files)} CID files")

        # Show folder structure
        print("\n📁 Folder structure:")
        print("  📂 Images folder:")
        for folder in sorted(created_image_folders):
            print(f"    └── {folder}/")

        print("  📂 Output folder:")
        for folder in sorted(created_image_folders):
            print(f"    └── {folder}/")
            # Show CID files in output folder
            output_folder_path_property = os.path.join(args.base_path, "output", folder)
            if os.path.exists(output_folder_path_property):
                cid_files = [f for f in os.listdir(output_folder_path_property) if f.endswith(".json")]
                for cid_file in cid_files:
                    print(f"        └── {cid_file}")

        print("\n🎉 Ready for image upload!")
        print(f"   Place images in: {os.path.join(args.base_path, 'images')}/[property_id]/")
        print(f"   CID files created in: {os.path.join(args.base_path, 'output')}/[property_id]/")
        print("   CID files created with actual IPFS CIDs")

    except Exception as e:
        print(f"❌ Folder creation failed: {e}")


if __name__ == "__main__":
    main()
