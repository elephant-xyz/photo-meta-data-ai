#!/usr/bin/env python3
"""
Simple IPFS traversal function
"""

import json
import requests
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def download_ipfs_data(cid: str, download_dir: str = "ipfs_downloads", max_depth: int = 10) -> Dict[str, any]:
    """
    Download all IPFS data from a given CID and traverse all relationships.
    
    Args:
        cid: The IPFS CID to start from
        download_dir: Directory to save downloaded files
        max_depth: Maximum depth for traversal
    
    Returns:
        Dictionary with traversal results
    """
    
    def fetch_from_ipfs(cid: str) -> Optional[Dict]:
        """Fetch content from IPFS"""
        try:
            url = f"https://ipfs.io/ipfs/{cid}"
            logger.info(f"Fetching: {cid}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching {cid}: {e}")
            return None
    
    def extract_cids(data: Dict) -> List[str]:
        """Extract CIDs from relationships"""
        cids = []
        
        def extract_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "/" and isinstance(value, str):
                        cids.append(value)
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)
        
        if "relationships" in data:
            extract_recursive(data["relationships"])
        
        # Check common CID fields
        for field in ["property_seed", "from", "to", "cid"]:
            if field in data:
                value = data[field]
                if isinstance(value, str):
                    cids.append(value)
                elif isinstance(value, dict) and "/" in value:
                    cids.append(value["/"])
        
        return list(set(cids))
    
    def save_file(cid: str, data: Dict, path: Path):
        """Save data to file"""
        file_path = path / f"{cid}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved: {file_path}")
    
    def traverse(cid: str, depth: int = 0) -> Dict:
        """Recursive traversal"""
        if depth >= max_depth:
            return {}
        
        logger.info(f"Traversing: {cid} (depth: {depth})")
        
        data = fetch_from_ipfs(cid)
        if not data:
            return {}
        
        # Save data
        download_path = Path(download_dir)
        download_path.mkdir(exist_ok=True)
        save_file(cid, data, download_path)
        
        # Find related CIDs
        related_cids = extract_cids(data)
        logger.info(f"Found {len(related_cids)} related CIDs")
        
        # Recursively traverse
        children = {}
        for related_cid in related_cids:
            child_result = traverse(related_cid, depth + 1)
            if child_result:
                children[related_cid] = child_result
        
        return {
            "cid": cid,
            "data": data,
            "related_cids": related_cids,
            "children": children
        }
    
    return traverse(cid)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download IPFS data")
    parser.add_argument("cid", help="CID to download")
    parser.add_argument("--download-dir", default="ipfs_downloads", help="Download directory")
    parser.add_argument("--max-depth", type=int, default=10, help="Max depth")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    result = download_ipfs_data(args.cid, args.download_dir, args.max_depth)
    print(f"Downloaded to: {args.download_dir}") 