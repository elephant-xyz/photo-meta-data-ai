#!/usr/bin/env python3
"""
IPFS Traverser - Downloads and traverses all IPFS data from a given CID
"""

import os
import json
import requests
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from urllib.parse import urljoin
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IPFSTraverser:
    """Traverses IPFS data and downloads all related content"""
    
    def __init__(self, base_url: str = "https://ipfs.io/ipfs/", download_dir: str = "ipfs_downloads"):
        self.base_url = base_url
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.downloaded_cids: Set[str] = set()
        self.failed_cids: Set[str] = set()
        self.relationships_found: Dict[str, List[str]] = {}
        
    def fetch_from_ipfs(self, cid: str) -> Optional[Dict]:
        """Fetch content from IPFS using the CID"""
        try:
            url = f"{self.base_url}{cid}"
            logger.info(f"Fetching: {cid}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching from IPFS CID {cid}: {e}")
            self.failed_cids.add(cid)
            return None
    
    def save_data(self, cid: str, data: Dict) -> str:
        """Save data to local file"""
        file_path = self.download_dir / f"{cid}.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Error saving {cid}: {e}")
            return ""
    
    def extract_cids_from_relationships(self, data: Dict) -> List[str]:
        """Extract all CIDs from relationships in the data"""
        cids = []
        
        def extract_cids_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "/" and isinstance(value, str):
                        # This is a CID reference
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
    
    def traverse_cid(self, cid: str, max_depth: int = 10, current_depth: int = 0) -> Dict[str, any]:
        """Recursively traverse a CID and all its relationships"""
        if current_depth >= max_depth:
            logger.warning(f"Max depth reached for CID: {cid}")
            return {}
        
        if cid in self.downloaded_cids:
            logger.info(f"Already downloaded: {cid}")
            return {}
        
        logger.info(f"Traversing CID: {cid} (depth: {current_depth})")
        
        # Fetch data from IPFS
        data = self.fetch_from_ipfs(cid)
        if not data:
            return {}
        
        # Save the data locally
        self.save_data(cid, data)
        self.downloaded_cids.add(cid)
        
        # Extract all CIDs from relationships
        related_cids = self.extract_cids_from_relationships(data)
        logger.info(f"Found {len(related_cids)} related CIDs: {related_cids}")
        
        # Store relationships
        self.relationships_found[cid] = related_cids
        
        # Recursively traverse each related CID
        results = {
            "cid": cid,
            "data": data,
            "related_cids": related_cids,
            "children": {}
        }
        
        for related_cid in related_cids:
            if related_cid not in self.downloaded_cids:
                child_result = self.traverse_cid(related_cid, max_depth, current_depth + 1)
                if child_result:
                    results["children"][related_cid] = child_result
        
        return results
    
    def traverse_from_file(self, file_path: str, max_depth: int = 10) -> Dict[str, any]:
        """Start traversal from a file containing a CID"""
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
            
            # Try to parse as JSON first
            try:
                data = json.loads(content)
                # Look for CIDs in the JSON
                cids = self.extract_cids_from_relationships(data)
                if cids:
                    logger.info(f"Found CIDs in JSON file: {cids}")
                    results = {}
                    for cid in cids:
                        results[cid] = self.traverse_cid(cid, max_depth)
                    return results
            except json.JSONDecodeError:
                pass
            
            # Treat as plain CID
            if content.startswith("bafkrei"):
                logger.info(f"Treating content as CID: {content}")
                return {content: self.traverse_cid(content, max_depth)}
            else:
                logger.error(f"Invalid CID format: {content}")
                return {}
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {}
    
    def generate_summary(self) -> Dict[str, any]:
        """Generate a summary of the traversal"""
        return {
            "total_downloaded": len(self.downloaded_cids),
            "total_failed": len(self.failed_cids),
            "downloaded_cids": list(self.downloaded_cids),
            "failed_cids": list(self.failed_cids),
            "relationships": self.relationships_found,
            "download_directory": str(self.download_dir)
        }
    
    def save_summary(self, filename: str = "ipfs_traversal_summary.json"):
        """Save traversal summary to file"""
        summary = self.generate_summary()
        summary_path = self.download_dir / filename
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary saved to: {summary_path}")
        except Exception as e:
            logger.error(f"Error saving summary: {e}")


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IPFS Traverser - Download all related IPFS data")
    parser.add_argument("input", help="CID or file path containing CID")
    parser.add_argument("--download-dir", default="ipfs_downloads", help="Download directory")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum traversal depth")
    parser.add_argument("--base-url", default="https://ipfs.io/ipfs/", help="IPFS gateway URL")
    parser.add_argument("--save-summary", action="store_true", help="Save traversal summary to file")
    
    args = parser.parse_args()
    
    # Create traverser
    traverser = IPFSTraverser(args.base_url, args.download_dir)
    
    # Check if input is a file or CID
    if os.path.exists(args.input):
        logger.info(f"Starting traversal from file: {args.input}")
        results = traverser.traverse_from_file(args.input, args.max_depth)
    else:
        logger.info(f"Starting traversal from CID: {args.input}")
        results = {args.input: traverser.traverse_cid(args.input, args.max_depth)}
    
    # Save summary if requested
    if args.save_summary:
        traverser.save_summary()
    
    # Print summary
    summary = traverser.generate_summary()
    logger.info("=" * 50)
    logger.info("TRAVERSAL SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total downloaded: {summary['total_downloaded']}")
    logger.info(f"Total failed: {summary['total_failed']}")
    logger.info(f"Download directory: {summary['download_directory']}")
    
    if summary['failed_cids']:
        logger.warning(f"Failed CIDs: {summary['failed_cids']}")


if __name__ == "__main__":
    main() 