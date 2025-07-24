#!/usr/bin/env python3
"""
Script to fix and submit local folder data:
1. Copy data to submit folder
2. Rename folders to propertyCID from upload results
3. Fix source_http_request
4. Run CLI command and run AI till data is fixed
"""

import os
import json
import shutil
import subprocess
import time
import csv
import openai
import requests
from pathlib import Path

# Load environment variables from .env file
def load_environment():
    """Load environment variables from .env file."""
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"📋 Loading environment variables from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Environment variables loaded")
    else:
        print(f"⚠️  No .env file found at {env_file}")

# Load environment variables
load_environment()

# IPFS Schema CIDs
IPFS_SCHEMA_CIDS = {
    "lot": "bafkreihjsl7r4nbuj4uipqiaejeaps55lsuitjzhl3yob266ejbekpkr6q",
    "layout": "bafkreiexvcm7ghuymwc3xigfk2jh5xhv4kqs5qngctck5hwkvgu4gl22w4",
    "structure": "bafkreid2wa56cecrm6ge4slmsm56xqy6j3gqlhldrljmruh64ams542xxe",
    "utility": "bafkreihuoaw6fm5abblivzgepkxkhduc5mivhho4rcidp5lvgb7fhyuide",
    "appliance": "bafkreieew4njulmeecnm3kah7w43eiali6lre5o45ttiyaqfjhb3ecu2mq",
    "file": "bafkreihug7qtvmblmpgdox7ex476inddz4u365gl33epmqoatiecqjveqq",
    "property": "bafkreih6x76aedhs7lqjk5uq4zskmfs33agku62b4flpq5s5pa6aek2gga"
}

# Relationship schema CID
RELATIONSHIP_SCHEMA_CID = "bafkreih226p5vjhx33jwgq7trblyplfw7yhkununuuahgpfok3hnh5mjwq"

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
            print(f"🔍 Trying to fetch schema from {gateway}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            schema_data = response.json()
            print(f"✅ Successfully fetched schema from {gateway}")
            return schema_data
        except Exception as e:
            print(f"⚠️ Error fetching from {gateway}: {e}")
            continue

    print(f"❌ Failed to fetch schema from IPFS CID {cid} from all gateways")
    return None

def load_upload_results():
    """Load upload results to get propertyCID mappings."""
    upload_results_file = "upload-results.csv"
    if not os.path.exists(upload_results_file):
        print(f"❌ Upload results file not found: {upload_results_file}")
        return {}
    
    property_cid_mapping = {}
    try:
        with open(upload_results_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = row.get('filePath', '')
                property_cid = row.get('propertyCid', '')
                
                if file_path and property_cid:
                    # Extract property ID from filePath like "/content/output/52434205310037080/..."
                    try:
                        parts = file_path.split('/')
                        for part in parts:
                            # Look for numeric property IDs (10+ digits)
                            if part.isdigit() and len(part) >= 10:
                                property_id = part
                                property_cid_mapping[property_id] = property_cid
                                print(f"📋 Found mapping: {property_id} -> {property_cid}")
                                break
                    except Exception as e:
                        print(f"⚠️  Could not extract property ID from {file_path}: {e}")
    except Exception as e:
        print(f"❌ Error reading upload results: {e}")
    
    return property_cid_mapping

def get_property_cid_from_upload_results():
    """Get property CIDs directly from upload results for folder naming."""
    upload_results_file = "upload-results.csv"
    if not os.path.exists(upload_results_file):
        print(f"❌ Upload results file not found: {upload_results_file}")
        return {}
    
    property_cids = {}
    try:
        with open(upload_results_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = row.get('filePath', '')
                property_cid = row.get('propertyCid', '')
                
                if file_path and property_cid:
                    # Extract property ID from filePath like "/content/output/52434205310037080/..."
                    try:
                        parts = file_path.split('/')
                        for part in parts:
                            # Look for numeric property IDs (10+ digits)
                            if part.isdigit() and len(part) >= 10:
                                property_id = part
                                property_cids[property_id] = property_cid
                                print(f"📋 Found property CID: {property_id} -> {property_cid}")
                                break
                    except Exception as e:
                        print(f"⚠️  Could not extract property ID from {file_path}: {e}")
    except Exception as e:
        print(f"❌ Error reading upload results: {e}")
    
    return property_cids

def get_parcel_id_from_property_json(folder_path):
    """Extract parcel ID from property.json file in a folder."""
    property_json_path = os.path.join(folder_path, "property.json")
    
    if not os.path.exists(property_json_path):
        print(f"⚠️  property.json not found in {folder_path}")
        return None
    
    try:
        with open(property_json_path, 'r') as f:
            data = json.load(f)
        
        # Try different possible field names for parcel ID
        parcel_id = (
            data.get('parcel_identifier') or 
            data.get('request_identifier') or
            data.get('parcelId')
        )
        
        if parcel_id:
            print(f"📋 Found parcel ID: {parcel_id} in {folder_path}")
            return parcel_id
        else:
            print(f"⚠️  No parcel ID found in property.json in {folder_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error reading property.json in {folder_path}: {e}")
        return None

def copy_data_to_submit_folder():
    """Copy data from output to submit folder."""
    output_dir = "output"
    submit_dir = "submit-photo"
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        return False
    
    # Create submit directory
    os.makedirs(submit_dir, exist_ok=True)
    
    # Copy all contents from output to submit
    try:
        for item in os.listdir(output_dir):
            src = os.path.join(output_dir, item)
            dst = os.path.join(submit_dir, item)
            
            if os.path.isdir(src):
                if os.path.exists(dst):
                    # Don't remove the entire directory, just copy files
                    print(f"📁 Directory {item} already exists, copying files...")
                    for root, dirs, files in os.walk(src):
                        # Calculate relative path
                        rel_path = os.path.relpath(root, src)
                        dst_root = os.path.join(dst, rel_path)
                        
                        # Create destination directories
                        for dir_name in dirs:
                            os.makedirs(os.path.join(dst_root, dir_name), exist_ok=True)
                        
                        # Copy files
                        for file_name in files:
                            src_file = os.path.join(root, file_name)
                            dst_file = os.path.join(dst_root, file_name)
                            shutil.copy2(src_file, dst_file)
                            print(f"  📄 Copied: {file_name}")
                else:
                    shutil.copytree(src, dst)
                    print(f"📁 Copied directory: {item}")
            else:
                shutil.copy2(src, dst)
                print(f"📄 Copied file: {item}")
        
        print(f"✅ Successfully copied data to {submit_dir}")
        return True
    except Exception as e:
        print(f"❌ Error copying data: {e}")
        return False

def rename_folders_to_property_cid(property_cid_mapping):
    """Rename folders to propertyCID from upload results."""
    submit_dir = "submit-photo"
    
    if not os.path.exists(submit_dir):
        print(f"❌ Submit directory not found: {submit_dir}")
        return False
    
    print(f"📋 Available property CID mappings: {list(property_cid_mapping.keys())}")
    
    renamed_count = 0
    for folder_name in os.listdir(submit_dir):
        folder_path = os.path.join(submit_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        print(f"🔍 Processing folder: {folder_name}")
        
        # Use folder name directly as parcel ID
        parcel_id = folder_name
        
        if parcel_id in property_cid_mapping:
            new_folder_name = property_cid_mapping[parcel_id]
            new_folder_path = os.path.join(submit_dir, new_folder_name)
            
            print(f"📋 Mapping: {parcel_id} -> {new_folder_name}")
            print(f"📁 Renaming: {folder_path} -> {new_folder_path}")
            
            if os.path.exists(new_folder_path):
                print(f"⚠️  Target folder already exists, removing: {new_folder_name}")
                shutil.rmtree(new_folder_path)
            
            if os.path.exists(folder_path):
                os.rename(folder_path, new_folder_path)
                print(f"🔄 Renamed: {folder_name} -> {new_folder_name} (parcel ID: {parcel_id})")
                renamed_count += 1
            else:
                print(f"❌ Source folder does not exist: {folder_path}")
        else:
            print(f"⚠️  No mapping found for folder: {folder_name} (parcel ID: {parcel_id})")
    
    print(f"✅ Renamed {renamed_count} folders")
    return True

def fix_source_http_request_in_file(file_path, property_id=None):
    """Fix source_http_request in a single file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        modified = False
        
        # Skip the main relationship file and CID files - never fix them
        filename = os.path.basename(file_path)
        if filename == "bafkreih226p5vjhx33jwgq7trblyplfw7yhkununuuahgpfok3hnh5mjwq.json" or (filename.endswith(".json") and filename.startswith("bafkre")):
            print(f"⏭️  Skipping CID file: {filename}")
            return False
        
        # Fix source_http_request (but not for relationship files)
        if not filename.startswith("relationship_"):
            # Add source_http_request if it doesn't exist
            if "source_http_request" not in data:
                data["source_http_request"] = {
                    "method": "GET",
                    "url": "https://pbcpao.gov/Property/Details"
                }
                modified = True
                print(f"  ➕ Added source_http_request")
            elif "source_http_request" in data:
                current_src = data["source_http_request"]
                
                # If it's empty or has wrong structure, fix it
                if not current_src or (isinstance(current_src, dict) and not current_src):
                    data["source_http_request"] = {
                        "method": "GET",
                        "url": "https://pbcpao.gov/Property/Details"
                    }
                    modified = True
                elif isinstance(current_src, dict):
                    # Ensure it has method and url, remove multiValueQueryString
                    if "method" not in current_src or "url" not in current_src:
                        data["source_http_request"] = {
                            "method": "GET",
                            "url": "https://pbcpao.gov/Property/Details"
                        }
                        modified = True
                    elif "multiValueQueryString" in current_src:
                        # Remove multiValueQueryString
                        del data["source_http_request"]["multiValueQueryString"]
                        modified = True
        
        # Fix request_identifier to use parcel_id if available
        if property_id and not filename.startswith("relationship_"):
            if "request_identifier" not in data:
                data["request_identifier"] = property_id
                modified = True
                print(f"  ➕ Added request_identifier: {property_id}")
            elif data["request_identifier"] != property_id:
                data["request_identifier"] = property_id
                modified = True
                print(f"  🔧 Updated request_identifier to parcel_id: {property_id}")
        
        # Ensure file document_type and file_format for file_xxxxxx files (but not relationship files)
        if "file_" in filename and "photo_metadata" in filename and not filename.startswith("relationship_"):
            if "document_type" not in data or data["document_type"] != "PropertyImage":
                data["document_type"] = "PropertyImage"
                modified = True
                print(f"  🔧 Set document_type to PropertyImage")
            
            # Fix file_format to be jpeg for photo metadata files
            if "file_format" not in data or data["file_format"] != "jpeg":
                data["file_format"] = "jpeg"
                modified = True
                print(f"  🔧 Set file_format to jpeg")
        
        # Ensure layout files have source_http_request (but not relationship files)
        if "layout" in filename and not filename.startswith("relationship_"):
            if "source_http_request" not in data:
                data["source_http_request"] = {
                    "method": "GET",
                    "url": "https://pbcpao.gov/Property/Details"
                }
                modified = True
                print(f"  ➕ Added source_http_request to layout file")
        
        if modified:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"🔧 Fixed source_http_request in: {os.path.basename(file_path)}")
            return True
        
        return False
    except Exception as e:
        print(f"❌ Error fixing file {file_path}: {e}")
        return False

def fix_source_http_request_in_directory(directory):
    """Fix source_http_request in all JSON files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return 0
    
    # Get property_id from property.json in this directory
    property_id = get_parcel_id_from_property_json(str(directory))
    
    fixed_count = 0
    
    # Find all JSON files
    for file_path in directory.rglob("*.json"):
        if file_path.is_file():
            if fix_source_http_request_in_file(str(file_path), property_id):
                fixed_count += 1
    
    print(f"✅ Fixed source_http_request and other issues in {fixed_count} files in {directory.name}")
    return fixed_count

def run_elephant_cli_validation(submit_dir, max_attempts=10):
    """Run Elephant CLI validation and return results."""
    try:
        # Run CLI validation from current directory (project root) using only submit folder
        cmd = [
            "npx", "@elephant-xyz/cli@latest", 
            "validate-and-upload", "submit-photo", 
            "--output-csv", "validation_results.csv"
        ]
        
        print(f"🔍 Running CLI validation: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ CLI validation completed successfully")
            return True, "validation_results.csv"
        else:
            # Check if it's a timeout error or HTTP 504 error
            stderr = result.stderr.lower()
            if ("timeout" in stderr or "timed out" in stderr or 
                "http" in stderr and "timeout" in stderr or
                "504" in stderr or "status: 504" in stderr):
                print("⏰ HTTP timeout/504 error detected - will retry without fixes")
                return "timeout", None
            else:
                print(f"⚠️  CLI validation failed: {result.stderr}")
                return False, None
            
    except subprocess.TimeoutExpired:
        print("⏰ CLI validation timed out")
        return "timeout", None
    except Exception as e:
        print(f"❌ Error running CLI validation: {e}")
        return False, None

def parse_validation_results(csv_file):
    """Parse validation results CSV to get error details."""
    if not os.path.exists(csv_file):
        return []
    
    errors = []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status', '').lower() == 'error':
                    errors.append({
                        'file': row.get('file', ''),
                        'error': row.get('error', ''),
                        'details': row
                    })
    except Exception as e:
        print(f"❌ Error parsing validation results: {e}")
    
    return errors

def check_submit_errors():
    """Check for submit errors in submit_errors.csv."""
    submit_errors_file = "submit_errors.csv"
    if not os.path.exists(submit_errors_file):
        return []
    
    errors = []
    try:
        with open(submit_errors_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                errors.append({
                    'file': row.get('file_path', ''),
                    'error': row.get('error_message', ''),
                    'path': row.get('path', ''),  # Add path information
                    'details': row
                })
    except Exception as e:
        print(f"❌ Error parsing submit errors: {e}")
    
    return errors

def trace_relationship_to_data_file(error_path, submit_dir):
    """Trace a relationship path to find the actual data file that needs fixing."""
    try:
        # Parse the error path to understand the relationship structure
        # Example: /relationships/property_has_layout/0/to
        path_parts = error_path.strip('/').split('/')
        
        if len(path_parts) >= 4 and path_parts[0] == 'relationships':
            relationship_type = path_parts[1]  # e.g., "property_has_layout"
            relationship_index = int(path_parts[2])  # e.g., 0
            relationship_end = path_parts[3]  # e.g., "to"
            
            # Find the main relationship file
            main_relationship_file = None
            for filename in os.listdir(submit_dir):
                if filename.startswith('bafkreih226p5vjhx33jwgq7trblyplfw7yhkununuuahgpfok3hnh5mjwq'):
                    main_relationship_file = os.path.join(submit_dir, filename)
                    break
            
            if not main_relationship_file:
                print(f"❌ Main relationship file not found in {submit_dir}")
                return None
            
            # Read the main relationship file
            with open(main_relationship_file, 'r') as f:
                main_relationships = json.load(f)
            
            # Get the relationship array
            if relationship_type not in main_relationships.get('relationships', {}):
                print(f"❌ Relationship type '{relationship_type}' not found")
                return None
            
            relationship_array = main_relationships['relationships'][relationship_type]
            
            if relationship_index >= len(relationship_array):
                print(f"❌ Relationship index {relationship_index} out of bounds")
                return None
            
            # Get the relationship file path
            relationship_file_path = relationship_array[relationship_index]['/']
            if relationship_file_path.startswith('./'):
                relationship_file_path = relationship_file_path[2:]  # Remove ./
            
            relationship_file_full_path = os.path.join(submit_dir, relationship_file_path)
            
            if not os.path.exists(relationship_file_full_path):
                print(f"❌ Relationship file not found: {relationship_file_full_path}")
                return None
            
            # Read the relationship file
            with open(relationship_file_full_path, 'r') as f:
                relationship_data = json.load(f)
            
            # Get the data file path based on relationship_end
            data_file_path = relationship_data.get(relationship_end, {}).get('/', '')
            if data_file_path.startswith('./'):
                data_file_path = data_file_path[2:]  # Remove ./
            
            data_file_full_path = os.path.join(submit_dir, data_file_path)
            
            if not os.path.exists(data_file_full_path):
                print(f"❌ Data file not found: {data_file_full_path}")
                return None
            
            print(f"🔍 Traced relationship: {error_path} -> {data_file_full_path}")
            return data_file_full_path
            
    except Exception as e:
        print(f"❌ Error tracing relationship: {e}")
        return None
    
    return None

def attempt_auto_fixes(submit_dir, errors):
    """Attempt to automatically fix common validation errors."""
    fixed_count = 0
    
    for error in errors:
        file_path = error.get('file', '')
        error_msg = error.get('error', '').lower()
        error_path = error.get('path', '')  # Get the specific path that failed
        
        if not file_path or not os.path.exists(file_path):
            continue
        
        # Skip the main relationship file - never fix it
        filename = os.path.basename(file_path)
        if filename == "bafkreibzrfmqka5h7dnuz7jzilgx4ht5rqcrx3ocl23nger65frbb5hzma.json":
            print(f"⏭️  Skipping main relationship file in auto-fixes: {filename}")
            continue
        
        # Check if this is a relationship path error
        actual_data_file = None
        if error_path and '/relationships/' in error_path:
            actual_data_file = trace_relationship_to_data_file(error_path, submit_dir)
        
        # Use the actual data file if found, otherwise use the original file
        target_file = actual_data_file if actual_data_file else file_path
        
        # Skip the main relationship file even if it's the target
        target_filename = os.path.basename(target_file)
        if target_filename == "bafkreibzrfmqka5h7dnuz7jzilgx4ht5rqcrx3ocl23nger65frbb5hzma.json":
            print(f"⏭️  Skipping main relationship file as target: {target_filename}")
            continue
        
        try:
            with open(target_file, 'r') as f:
                data = json.load(f)
            
            modified = False
            
            # Get property_id from the directory containing this file
            property_dir = os.path.dirname(target_file)
            property_id = get_parcel_id_from_property_json(property_dir)
            
            # Get schema for this file type
            schema = get_schema_for_file(target_file)
            
            # Ensure all schema properties are present (but not for relationship files)
            # Check if this is a relationship file by filename or content structure
            filename = os.path.basename(target_file).lower()
            is_relationship_file = (
                "relationship" in filename or 
                filename.startswith("bafkreicaq62gggwbppihgstao2maakafmghjttf73ai53tz5tam2cixrvu") or
                (isinstance(data, dict) and "relationships" in data and "label" in data)
            )
            if schema:
                data, schema_modified = ensure_all_schema_properties(data, schema, is_relationship_file)
                if schema_modified:
                    modified = True
            
            # Fix common issues
            if "source_http_request" in error_msg and not filename.startswith("relationship_"):
                # Fix source_http_request
                if "source_http_request" in data:
                    data["source_http_request"] = {
                        "method": "GET",
                        "url": "https://pbcpao.gov/Property/Details"
                    }
                    modified = True
            
            elif "missing required" in error_msg and not filename.startswith("relationship_"):
                # Add missing required fields as null
                if "source_http_request" not in data:
                    data["source_http_request"] = {
                        "method": "GET",
                        "url": "https://pbcpao.gov/Property/Details"
                    }
                    modified = True
            
            # Fix request_identifier to use parcel_id if available
            if property_id and "request_identifier" in data:
                if data["request_identifier"] != property_id:
                    data["request_identifier"] = property_id
                    modified = True
                    print(f"  🔧 Updated request_identifier to parcel_id: {property_id}")
            
            # Ensure file document_type and file_format for file_xxxxxx files (but not relationship files)
            if "file_" in filename and "photo_metadata" in filename and not filename.startswith("relationship_"):
                if "document_type" not in data or data["document_type"] != "PropertyImage":
                    data["document_type"] = "PropertyImage"
                    modified = True
                    print(f"  🔧 Set document_type to PropertyImage")
                
                # Fix file_format to be jpeg for photo metadata files
                if "file_format" not in data or data["file_format"] != "jpeg":
                    data["file_format"] = "jpeg"
                    modified = True
                    print(f"  🔧 Set file_format to jpeg")
            
            # Ensure layout files have source_http_request (but not relationship files)
            if "layout" in filename and not filename.startswith("relationship_"):
                if "source_http_request" not in data:
                    data["source_http_request"] = {
                        "method": "GET",
                        "url": "https://pbcpao.gov/Property/Details"
                    }
                    modified = True
                    print(f"  ➕ Added source_http_request to layout file")
            
            # Fix string or null fields (convert arrays to strings)
            data, string_fixed = fix_string_or_null_fields(data)
            if string_fixed:
                modified = True
            
            elif "invalid json" in error_msg:
                # Try to fix JSON syntax
                # This is a basic fix - might need more sophisticated handling
                pass
            
            if modified:
                with open(target_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"🔧 Auto-fixed: {os.path.basename(target_file)}")
                fixed_count += 1
                
        except Exception as e:
            print(f"❌ Error auto-fixing {target_file}: {e}")
    
    return fixed_count

def ensure_layout_files_have_required_fields(submit_dir):
    """Ensure all layout files have required fields before CLI validation."""
    fixed_count = 0
    
    for root, dirs, files in os.walk(submit_dir):
        for file in files:
            if file.endswith('.json') and 'layout' in file.lower():
                file_path = os.path.join(root, file)
                
                # Skip the main relationship file and all relationship files
                if file == "bafkreibzrfmqka5h7dnuz7jzilgx4ht5rqcrx3ocl23nger65frbb5hzma.json" or file.startswith("relationship_"):
                    continue
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    modified = False
                    
                    # Get property_id from the directory containing this file
                    property_dir = os.path.dirname(file_path)
                    property_id = get_parcel_id_from_property_json(property_dir)
                    
                    # Ensure source_http_request is present
                    if "source_http_request" not in data:
                        data["source_http_request"] = {
                            "method": "GET",
                            "url": "https://pbcpao.gov/Property/Details"
                        }
                        modified = True
                        print(f"  ➕ Added source_http_request to {file}")
                    
                    # Ensure request_identifier is present with parcel_id
                    if property_id and ("request_identifier" not in data or data["request_identifier"] != property_id):
                        data["request_identifier"] = property_id
                        modified = True
                        print(f"  🔧 Updated request_identifier to {property_id} in {file}")
                    
                    # Add missing required fields for ALL layout files (including pool fields and layout fields)
                    all_layout_fields = [
                        # Pool fields
                        "pool_type", "pool_equipment", "spa_type", "safety_features",
                        "pool_condition", "pool_surface_type", "pool_water_quality",
                        # Layout fields
                        "flooring_material_type", "floor_level", "window_design_type", 
                        "window_material_type", "window_treatment_type", "paint_condition",
                        "flooring_wear", "clutter_level", "visible_damage", 
                        "fixture_finish_quality", "design_style", "decor_elements",
                        "view_type", "condition_issues"
                    ]
                    for field in all_layout_fields:
                        if field not in data:
                            data[field] = None
                            modified = True
                            print(f"  ➕ Added missing layout field '{field}' to {file}")
                    
                    # Fix string or null fields (convert arrays to strings)
                    data, string_fixed = fix_string_or_null_fields(data)
                    if string_fixed:
                        modified = True
                    
                    if modified:
                        with open(file_path, 'w') as f:
                            json.dump(data, f, indent=2)
                        fixed_count += 1
                        
                except Exception as e:
                    print(f"❌ Error fixing layout file {file}: {e}")
    
    return fixed_count

def fix_string_or_null_fields(data):
    """Fix fields that should be string or null but are currently arrays."""
    modified = False
    
    # Fields that should be string or null, not arrays
    string_or_null_fields = [
        "decor_elements", "view_type", "condition_issues", "design_style",
        "visible_damage", "clutter_level", "paint_condition", "flooring_wear",
        "fixture_finish_quality", "window_treatment_type", "window_material_type",
        "window_design_type", "flooring_material_type", "pool_type", "pool_equipment",
        "spa_type", "safety_features", "pool_condition", "pool_surface_type",
        "pool_water_quality"
    ]
    
    for field in string_or_null_fields:
        if field in data:
            current_value = data[field]
            # If it's an array, convert to string
            if isinstance(current_value, list):
                if len(current_value) == 0:
                    data[field] = None
                else:
                    data[field] = str(current_value[0]) if current_value[0] is not None else None
                modified = True
                print(f"  🔧 Converted {field} from array to string/null")
    
    return data, modified

def ensure_all_schema_properties(data, schema, is_relationship_file=False):
    """Ensure all schema properties are present in data, adding null for missing ones."""
    if not schema or "properties" not in schema:
        return data, False
    
    # Don't add data properties to relationship files
    if is_relationship_file:
        return data, False
    
    properties = schema["properties"]
    modified = False
    
    for prop_name, prop_schema in properties.items():
        if prop_name not in data:
            # Add missing property with appropriate null value based on type
            prop_type = prop_schema.get("type", "string")
            if prop_type == "array":
                data[prop_name] = []
            elif prop_type == "object":
                data[prop_name] = {}
            elif prop_type == "number":
                data[prop_name] = None
            elif prop_type == "boolean":
                data[prop_name] = None
            else:
                data[prop_name] = None
            modified = True
            print(f"  ➕ Added missing property '{prop_name}' with null value")
    
    return data, modified

def get_schema_for_file(file_path):
    """Get the schema information for a file based on its type from IPFS."""
    filename = os.path.basename(file_path).lower()
    
    # Determine schema type based on filename
    schema_type = None
    schema_cid = None
    
    if "property.json" in filename:
        schema_type = "property"
        schema_cid = IPFS_SCHEMA_CIDS["property"]
    elif "photo_metadata" in filename or "file_" in filename:
        schema_type = "file"
        schema_cid = IPFS_SCHEMA_CIDS["file"]
    elif "relationship" in filename:
        schema_type = "relationship"
        schema_cid = RELATIONSHIP_SCHEMA_CID
    elif "layout" in filename:
        schema_type = "layout"
        schema_cid = IPFS_SCHEMA_CIDS["layout"]
    elif "structure" in filename:
        schema_type = "structure"
        schema_cid = IPFS_SCHEMA_CIDS["structure"]
    elif "lot" in filename:
        schema_type = "lot"
        schema_cid = IPFS_SCHEMA_CIDS["lot"]
    elif "utility" in filename:
        schema_type = "utility"
        schema_cid = IPFS_SCHEMA_CIDS["utility"]
    elif "appliance" in filename:
        schema_type = "appliance"
        schema_cid = IPFS_SCHEMA_CIDS["appliance"]
    else:
        # Default to file schema for unknown types
        schema_type = "file"
        schema_cid = IPFS_SCHEMA_CIDS["file"]
    
    print(f"📋 Fetching {schema_type} schema for {filename}")
    
    # Fetch schema from IPFS
    schema = fetch_schema_from_ipfs(schema_cid)
    if schema:
        print(f"✅ Retrieved {schema_type} schema from IPFS")
        
        # Keep all schema properties - they will be added with null values if missing
        
        return schema
    else:
        print(f"⚠️ Failed to fetch {schema_type} schema, using fallback")
        # Fallback to basic schema
        return {
            "type": "object",
            "properties": {
                "source_http_request": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string"},
                        "url": {"type": "string"}
                    },
                    "required": ["method", "url"]
                }
            }
        }

def fix_errors_with_openai(submit_dir, errors):
    """Use OpenAI to fix validation errors."""
    fixed_count = 0
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("📋 Please ensure your .env file contains: OPENAI_API_KEY=your-api-key-here")
        return 0
    
    # Set up OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key)
        print(f"✅ OpenAI client initialized with API key: {api_key[:10]}...")
    except Exception as e:
        print(f"❌ Error setting up OpenAI client: {e}")
        return 0
    
    # Process errors in batches to speed up
    batch_size = 10  # Process 10 errors at a time
    for i in range(0, len(errors), batch_size):
        batch = errors[i:i + batch_size]
        print(f"🤖 Processing batch {i//batch_size + 1}/{(len(errors) + batch_size - 1)//batch_size} ({len(batch)} errors)")
        
        for error in batch:
            file_path = error.get('file', '')
            error_msg = error.get('error', '')
            error_path = error.get('path', '')  # Get the specific path that failed
            
            if not file_path or not os.path.exists(file_path):
                continue
            
            # Check if this is a relationship path error
            actual_data_file = None
            if error_path and '/relationships/' in error_path:
                actual_data_file = trace_relationship_to_data_file(error_path, submit_dir)
            
            # Use the actual data file if found, otherwise use the original file
            target_file = actual_data_file if actual_data_file else file_path
            
            try:
                # Read the current file content
                with open(target_file, 'r') as f:
                    current_content = f.read()
                
                # Get schema for this file type
                schema = get_schema_for_file(target_file)
                
                # Create detailed prompt for OpenAI
                filename = os.path.basename(target_file).lower()
                layout_specific_instruction = ""
                if "layout" in filename:
                    layout_specific_instruction = "\n- For layout files, ALWAYS include source_http_request with method 'GET' and url 'https://pbcpao.gov/Property/Details'"
                
                prompt = f"""
You are a JSON validation expert working with Elephant Network schemas. Please fix the JSON file to resolve the validation error.

CONTEXT:
- File: {os.path.basename(target_file)}
- Error: {error_msg}
- This is for Elephant Network data validation

SCHEMA REQUIREMENTS:
{json.dumps(schema, indent=2)}

CURRENT JSON CONTENT:
{current_content}

INSTRUCTIONS:
1. Analyze the error message and current JSON
2. Fix the JSON to match the schema requirements
3. Ensure ALL schema properties are present (add null for missing ones)
4. Maintain data integrity while fixing validation issues
5. Return ONLY the corrected JSON, no explanations or markdown

IMPORTANT RULES:
- ALL schema properties must be present - add null for any missing properties
- Always include source_http_request with method "GET" and url "https://pbcpao.gov/Property/Details"{layout_specific_instruction}
- For relationship files, ensure "from" and "to" objects have "/" keys pointing to file paths
- For photo metadata files, ensure proper file_path and file_name fields
- For property files, ensure parcel_identifier and other required fields are present
- Never remove properties - only add missing ones with null values

Please provide the corrected JSON:
"""
                
                print(f"  🤖 Fixing: {os.path.basename(target_file)}")
                
                # Call OpenAI with shorter timeout
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a JSON validation expert specializing in Elephant Network schemas. Always return valid JSON that matches the provided schema requirements."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    timeout=30  # 30 second timeout per request
                )
                
                # Extract the fixed JSON
                fixed_json = response.choices[0].message.content.strip()
                
                # Clean up the response (remove markdown if present)
                if fixed_json.startswith("```json"):
                    fixed_json = fixed_json.split("```json")[1]
                if fixed_json.endswith("```"):
                    fixed_json = fixed_json.rsplit("```", 1)[0]
                fixed_json = fixed_json.strip()
                
                # Try to parse the fixed JSON to validate it
                try:
                    fixed_data = json.loads(fixed_json)
                    
                    # Ensure all schema properties are present in the OpenAI-fixed data (but not for relationship files)
                    filename = os.path.basename(target_file).lower()
                    is_relationship_file = (
                        "relationship" in filename or 
                        filename.startswith("bafkreicaq62gggwbppihgstao2maakafmghjttf73ai53tz5tam2cixrvu") or
                        (isinstance(fixed_data, dict) and "relationships" in fixed_data and "label" in fixed_data)
                    )
                    schema = get_schema_for_file(target_file)
                    if schema:
                        fixed_data, schema_modified = ensure_all_schema_properties(fixed_data, schema, is_relationship_file)
                        if schema_modified:
                            print(f"    ➕ Added missing schema properties")
                    
                    # Ensure layout files always have source_http_request
                    if "layout" in filename and "source_http_request" not in fixed_data:
                        fixed_data["source_http_request"] = {
                            "method": "GET",
                            "url": "https://pbcpao.gov/Property/Details"
                        }
                        print(f"    ➕ Added source_http_request to layout file")
                    
                    # Write the fixed JSON back to the file
                    with open(target_file, 'w') as f:
                        json.dump(fixed_data, f, indent=2)
                    
                    print(f"    ✅ Fixed: {os.path.basename(target_file)}")
                    fixed_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"    ❌ Invalid JSON for {os.path.basename(file_path)}: {e}")
                    
            except Exception as e:
                print(f"    ❌ Error fixing {target_file}: {e}")
        
        # Small delay between batches to avoid rate limits
        if i + batch_size < len(errors):
            time.sleep(1)
    
    return fixed_count

def copy_property_files_from_zip():
    """Unzip submit.zip, find property.json files, and copy to submit-photo property folders."""
    import zipfile
    import shutil
    
    if not os.path.exists("submit.zip"):
        print("⚠️  submit.zip not found - skipping property.json copy")
        return
    
    print("📦 Unzipping submit.zip...")
    
    try:
        # Create temporary extraction directory
        extract_dir = "submit-extracted"
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        
        # Extract the zip file
        with zipfile.ZipFile("submit.zip", "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print("✅ Successfully unzipped submit.zip")
        
        # Find all property.json files in the extracted content
        property_files = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file == "property.json":
                    property_files.append(os.path.join(root, file))
        
        print(f"📁 Found {len(property_files)} property.json files in submit.zip")
        
        # Copy property.json files to corresponding submit-photo folders
        copied_count = 0
        for property_file in property_files:
            # Extract the property CID from the path
            # Path format: submit-extracted/submit/bafkreixxx/property.json
            path_parts = property_file.split(os.sep)
            if len(path_parts) >= 4:
                property_cid = path_parts[2]  # The CID folder name from the zip (after 'submit')
                
                # Target path in submit-photo
                target_dir = os.path.join("submit-photo", property_cid)
                target_file = os.path.join(target_dir, "property.json")
                
                # Create target directory if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)
                
                # Copy the property.json file
                try:
                    shutil.copy2(property_file, target_file)
                    print(f"✅ Copied: {property_cid}/property.json")
                    copied_count += 1
                except Exception as e:
                    print(f"❌ Error copying {property_cid}/property.json: {e}")
        
        print(f"✅ Successfully copied {copied_count} property.json files to submit-photo folders")
        
        # Clean up extracted files
        shutil.rmtree(extract_dir)
        print("🧹 Cleaned up extracted files")
        
    except Exception as e:
        print(f"❌ Error processing submit.zip: {e}")
        # Clean up on error
        if os.path.exists("submit-extracted"):
            shutil.rmtree("submit-extracted")


def main():
    """Main function to orchestrate the entire process."""
    print("🚀 Starting fix and submit process for local folders...")
    
    # Step 1: Load property CIDs from upload results
    print("\n📋 Step 1: Loading property CIDs from upload results...")
    property_cids = get_property_cid_from_upload_results()
    if not property_cids:
        print("❌ No property CIDs found. Please ensure upload-results.csv exists.")
        return
    
    # Step 2: Copy data to submit folder
    print("\n📁 Step 2: Copying data to submit folder...")
    if not copy_data_to_submit_folder():
        print("❌ Failed to copy data to submit folder")
        return
    
    # Step 3: Fix source_http_request and other issues in all files
    print("\n🔧 Step 3: Fixing source_http_request and other issues...")
    submit_dir = "submit-photo"
    total_fixed = 0
    
    # Fix files in submit folder
    for folder_name in os.listdir(submit_dir):
        folder_path = os.path.join(submit_dir, folder_name)
        if os.path.isdir(folder_path):
            fixed_count = fix_source_http_request_in_directory(folder_path)
            total_fixed += fixed_count
    
    print(f"✅ Total files fixed: {total_fixed}")
    
    # Step 4: Ensure all layout files have required fields before CLI
    print("\n🔧 Step 4: Ensuring all layout files have required fields...")
    submit_dir = "submit"
    layout_fixed_count = ensure_layout_files_have_required_fields(submit_dir)
    print(f"✅ Fixed {layout_fixed_count} layout files")
    
    # Step 5: Rename folders to propertyCID
    print("\n🔄 Step 5: Renaming folders to propertyCID...")
    if not rename_folders_to_property_cid(property_cids):
        print("❌ Failed to rename folders")
        return
    
    # Step 6: Copy property.json files from submit.zip after folder renaming
    print("\n📋 Step 6: Copying property.json files from submit.zip...")
    copy_property_files_from_zip()
    
    # Step 7: Run CLI validation iteratively until no submit errors
    print("\n🔍 Step 7: Running CLI validation iteratively until no submit errors...")
    max_attempts = 50  # Increased max attempts
    attempt = 0
    no_errors_found = False
    
    while attempt < max_attempts and not no_errors_found:
        attempt += 1
        print(f"\n🔄 Attempt {attempt}/{max_attempts}")
        
        # Run CLI validation
        success, csv_file = run_elephant_cli_validation(submit_dir)
        
        if success == "timeout":
            print("⏰ HTTP timeout detected - checking for 504 errors in submit_errors.csv...")
            
            # Check if there are 504 errors in submit_errors.csv
            submit_errors = check_submit_errors()
            has_504_errors = any("504" in error.get('error', '').lower() for error in submit_errors)
            
            if has_504_errors:
                print("⏰ Found 504 errors (IPFS gateway timeouts) - retrying CLI immediately without fixes...")
                print("🔄 504 errors are network issues, not data issues - no fixes needed")
            else:
                print("⏰ No 504 errors found in submit_errors.csv - retrying without fixes...")
            
            # Wait a bit longer before retry for timeout
            time.sleep(5)
            continue
        elif success:
            # Check for submit errors (more important than validation results)
            submit_errors = check_submit_errors()
            
            if not submit_errors:
                print("🎉 No submit errors found! Validation successful!")
                no_errors_found = True
                break
            else:
                print(f"⚠️  Found {len(submit_errors)} submit errors")
                
                # Also check validation results for additional fixes
                validation_errors = parse_validation_results(csv_file)
                if validation_errors:
                    print(f"⚠️  Found {len(validation_errors)} validation errors")
                
                # Attempt auto-fixes for both types of errors
                all_errors = submit_errors + validation_errors
                fixed_count = attempt_auto_fixes(submit_dir, all_errors)
                
                if fixed_count == 0:
                    print("🤖 No auto-fixes applied. Trying OpenAI to fix errors...")
                    openai_fixed_count = fix_errors_with_openai(submit_dir, all_errors)
                    
                    if openai_fixed_count == 0:
                        print("⚠️  OpenAI couldn't fix errors. Manual intervention may be needed.")
                        break
                    else:
                        print(f"🤖 OpenAI applied {openai_fixed_count} fixes")
                else:
                    print(f"🔧 Applied {fixed_count} auto-fixes")
        else:
            print("❌ CLI validation failed")
            break
        
        # Wait a bit before next attempt
        time.sleep(2)
    
    if attempt >= max_attempts:
        print("⚠️  Reached maximum attempts. Some issues may remain.")
    elif no_errors_found:
        print("🎉 Successfully resolved all submit errors!")
    
    print("\n✅ Fix and submit process completed!")
    print(f"📁 Submit folder: {submit_dir}")
    print("🔍 Check validation_results.csv for final results")

if __name__ == "__main__":
    main() 