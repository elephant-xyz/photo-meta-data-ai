#!/usr/bin/env python3
"""
Script to retrieve the layout schema from IPFS using the CID from the fix script.
"""

import json
import requests

# Layout schema CID from fix script
LAYOUT_SCHEMA_CID = "bafkreiexvcm7ghuymwc3xigfk2jh5xhv4kqs5qngctck5hwkvgu4gl22w4"

def fetch_schema_from_ipfs(cid):
    """Fetch schema from IPFS using the provided CID."""
    gateways = [
        "https://ipfs.io/ipfs/",
        "https://gateway.pinata.cloud/ipfs/",
        "https://cloudflare-ipfs.com/ipfs/",
        "https://dweb.link/ipfs/",
        "https://ipfs.infura.io/ipfs/"
    ]

    for gateway in gateways:
        try:
            url = f"{gateway}{cid}"
            print(f"🔍 Trying to fetch layout schema from {gateway}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            schema_data = response.json()
            print(f"✅ Successfully fetched layout schema from {gateway}")
            return schema_data
        except Exception as e:
            print(f"⚠️ Error fetching from {gateway}: {e}")
            continue

    print(f"❌ Failed to fetch layout schema from IPFS CID {cid} from all gateways")
    return None

def save_schema_to_file(schema_data, filename="layout_schema.json"):
    """Save the schema data to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(schema_data, f, indent=2)
        print(f"💾 Layout schema saved to: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving schema to file: {e}")
        return False

def main():
    """Main function to retrieve and save the layout schema."""
    print("🔍 Retrieving layout schema from IPFS...")
    print(f"📋 Layout Schema CID: {LAYOUT_SCHEMA_CID}")
    
    # Fetch the schema
    schema_data = fetch_schema_from_ipfs(LAYOUT_SCHEMA_CID)
    
    if schema_data:
        print("\n📄 Layout Schema Content:")
        print(json.dumps(schema_data, indent=2))
        
        # Save to file
        save_schema_to_file(schema_data)
        
        print("\n✅ Layout schema retrieved successfully!")
    else:
        print("\n❌ Failed to retrieve layout schema")

if __name__ == "__main__":
    main() 