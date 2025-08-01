#!/usr/bin/env python3
"""
Simple IPFS Traverser - Downloads and traverses all IPFS data from a given CID
"""

import os
import json
import requests
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_from_ipfs(cid: str, base_url: str = "https://ipfs.io/ipfs/") -> Optional[Dict]:
    """Fetch content from IPFS using the CID"""
    try:
        url = f"{base_url}{cid}"
        logger.info(f"Fetching: {cid}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching from IPFS CID {cid}: {e}")
        return None


def extract_cids_from_relationships(data: Dict) -> List[str]:
    """Extract all CIDs from relationships in the data"""
    cids = []
    
    def extract_cids_recursive(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "/" and isinstance(value, str):
                    cids.append(value)
                elif isinstance(value, (dict, list)):
                    extract_cids_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_cids_recursive(item)
    
    # Check for relationships field
    if "relationships" in data:
        extract_cids_recursive(data["relationships"])
    
    # Check for other common CID fields
    for field in ["property_seed", "from", "to", "cid", "ipfs"]:
        if field in data:
            value = data[field]
            if isinstance(value, str):
                cids.append(value)
            elif isinstance(value, dict) and "/" in value:
                cids.append(value["/"])
    
    # Check for any field that looks like a CID (starts with bafkrei)
    for key, value in data.items():
        if isinstance(value, str) and value.startswith("bafkrei"):
            cids.append(value)
        elif isinstance(value, dict) and "/" in value:
            cid_value = value["/"]
            if isinstance(cid_value, str) and cid_value.startswith("bafkrei"):
                cids.append(cid_value)
    
    return list(set(cids))  # Remove duplicates


def save_data(cid: str, data: Dict, download_dir: Path) -> str:
    """Save data to local file"""
    file_path = download_dir / f"{cid}.json"
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving {cid}: {e}")
        return ""


def traverse_ipfs(cid: str, download_dir: str = "ipfs_downloads", max_depth: int = 10, current_depth: int = 0) -> Dict[str, any]:
    """Recursively traverse a CID and all its relationships"""
    download_path = Path(download_dir)
    download_path.mkdir(exist_ok=True)
    
    if current_depth >= max_depth:
        logger.warning(f"Max depth reached for CID: {cid}")
        return {}
    
    logger.info(f"Traversing CID: {cid} (depth: {current_depth})")
    
    # Fetch data from IPFS
    data = fetch_from_ipfs(cid)
    if not data:
        return {}
    
    # Save the data locally
    save_data(cid, data, download_path)
    
    # Extract all CIDs from relationships
    related_cids = extract_cids_from_relationships(data)
    logger.info(f"Found {len(related_cids)} related CIDs: {related_cids}")
    
    # Recursively traverse each related CID
    results = {
        "cid": cid,
        "data": data,
        "related_cids": related_cids,
        "children": {}
    }
    
    for related_cid in related_cids:
        child_result = traverse_ipfs(related_cid, download_dir, max_depth, current_depth + 1)
        if child_result:
            results["children"][related_cid] = child_result
    
    return results


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple IPFS Traverser - Download all related IPFS data")
    parser.add_argument("cid", help="CID to traverse")
    parser.add_argument("--download-dir", default="ipfs_downloads", help="Download directory")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum traversal depth")
    parser.add_argument("--base-url", default="https://ipfs.io/ipfs/", help="IPFS gateway URL")
    
    args = parser.parse_args()
    
    logger.info(f"Starting traversal of CID: {args.cid}")
    results = traverse_ipfs(args.cid, args.download_dir, args.max_depth)
    
    logger.info("=" * 50)
    logger.info("TRAVERSAL COMPLETED")
    logger.info("=" * 50)
    logger.info(f"Download directory: {args.download_dir}")


if __name__ == "__main__":
    main() 