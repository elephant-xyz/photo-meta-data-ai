#!/usr/bin/env python3
"""
Script to get property data from IPFS and create folder structure
"""

import os
import json
import requests
import sys
import pandas as pd
from pathlib import Path

def fetch_from_ipfs(cid):
    """Fetch data from IPFS CID"""
    print(f"🔍 Fetching from IPFS: {cid}")
    
    try:
        response = requests.get(f"https://ipfs.io/ipfs/{cid}", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Successfully fetched data from {cid}")
            return data
        else:
            print(f"❌ Failed to fetch from IPFS: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error fetching from IPFS: {e}")
        return None

def get_first_property_cid_from_upload_results():
    """Get the first property CID from upload_results.csv"""
    print("📊 Reading upload_results.csv...")
    
    csv_file = "upload_results.csv"
    if not os.path.exists(csv_file):
        print(f"❌ {csv_file} not found!")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} records from {csv_file}")
        
        if len(df) == 0:
            print("❌ No records found in upload_results.csv")
            return None
        
        # Get the first property CID
        first_property_cid = df.iloc[0]['propertyCid']
        print(f"📋 First property CID: {first_property_cid}")
        
        return first_property_cid
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

def process_property_cid(property_cid):
    """Process a single property CID"""
    print(f"\n🏠 Processing property CID: {property_cid}")
    
    # Step 1: Fetch the property data from IPFS
    property_data = fetch_from_ipfs(property_cid)
    if not property_data:
        print(f"❌ Failed to fetch property data for {property_cid}")
        return None
    
    # Step 2: Look for property_seed relationship
    relationships = property_data.get('relationships', {})
    property_seed_relationship = relationships.get('property_seed')
    
    if not property_seed_relationship:
        print(f"❌ No property_seed relationship found in {property_cid}")
        return None
    
    # Extract the CID from the relationship
    property_seed_cid = property_seed_relationship.get('/', '')
    if not property_seed_cid:
        print(f"❌ No CID found in property_seed relationship")
        return None
    
    print(f"🔗 Found property_seed CID: {property_seed_cid}")
    
    # Step 3: Fetch the property_seed data
    property_seed_data = fetch_from_ipfs(property_seed_cid)
    if not property_seed_data:
        print(f"❌ Failed to fetch property_seed data for {property_seed_cid}")
        return None
    
    # Step 4: Extract parcel_id
    parcel_id = property_seed_data.get('parcel_id')
    if not parcel_id:
        print(f"❌ No parcel_id found in property_seed data")
        return None
    
    print(f"📦 Found parcel_id: {parcel_id}")
    
    # Step 5: Create output directory
    output_dir = os.path.join("output", str(parcel_id))
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created directory: {output_dir}")
    
    # Step 6: Save the property_seed data as CID.json
    cid_filename = f"{property_seed_cid}.json"
    cid_filepath = os.path.join(output_dir, cid_filename)
    
    with open(cid_filepath, 'w') as f:
        json.dump(property_seed_data, f, indent=2)
    
    print(f"💾 Saved {cid_filename} to {output_dir}")
    
    return {
        'property_cid': property_cid,
        'property_seed_cid': property_seed_cid,
        'parcel_id': parcel_id,
        'output_dir': output_dir,
        'cid_file': cid_filepath
    }

def process_specific_ipfs_url(ipfs_url):
    """Process a specific IPFS URL"""
    print(f"🔍 Processing specific IPFS URL: {ipfs_url}")
    
    # Extract CID from URL
    if '/ipfs/' in ipfs_url:
        cid = ipfs_url.split('/ipfs/')[-1]
    else:
        print(f"❌ Invalid IPFS URL format: {ipfs_url}")
        return None
    
    print(f"📋 Extracted CID: {cid}")
    
    # Process this CID
    return process_property_cid(cid)

def main():
    """Main function"""
    print("🚀 Property IPFS Processing Script")
    print("=" * 40)
    
    # Check if specific IPFS URL is provided
    if len(sys.argv) > 1:
        ipfs_url = sys.argv[1]
        print(f"🎯 Processing specific IPFS URL: {ipfs_url}")
        
        result = process_specific_ipfs_url(ipfs_url)
        if result:
            print(f"\n✅ Successfully processed {ipfs_url}")
            print(f"📁 Output: {result['output_dir']}")
            print(f"💾 File: {result['cid_file']}")
        else:
            print(f"❌ Failed to process {ipfs_url}")
        return
    
    # Get the first property CID from upload_results.csv
    print("📊 Getting first property CID from upload_results.csv")
    
    property_cid = get_first_property_cid_from_upload_results()
    if not property_cid:
        print("❌ No property CID found")
        return
    
    # Process the first property CID
    result = process_property_cid(property_cid)
    if result:
        print(f"\n✅ Successfully processed first property CID")
        print(f"📁 Output: {result['output_dir']}")
        print(f"💾 File: {result['cid_file']}")
        
        # Create summary file
        summary_file = "ipfs_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump([result], f, indent=2)
        
        print(f"📊 Summary saved to: {summary_file}")
    else:
        print(f"❌ Failed to process property CID: {property_cid}")

if __name__ == "__main__":
    main() 