#!/usr/bin/env python3
"""
Schema Validation Fixer Script

This script processes AI output data, retrieves schemas from IPFS, and fixes schema validation issues:
- Sets invalid enum values to null
- Converts string booleans to actual booleans
- Ensures required fields are present
- Fixes data type mismatches
"""

import os
import json
import requests
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fix-schema-validation.log')
        # Removed StreamHandler to only log to files
    ]
)
logger = logging.getLogger(__name__)

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

# IPFS Gateways
IPFS_GATEWAYS = [
    "https://ipfs.io/ipfs/",
    "https://gateway.pinata.cloud/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
    "https://dweb.link/ipfs/",
    "https://gateway.ipfs.io/ipfs/"
]

def fetch_schema_from_ipfs(cid: str) -> Optional[Dict[str, Any]]:
    """Fetch schema from IPFS using multiple gateways."""
    for gateway in IPFS_GATEWAYS:
        try:
            url = f"{gateway}{cid}"
            logger.info(f"Fetching schema from {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            schema = response.json()
            logger.info(f"✅ Successfully fetched schema from {gateway}")
            return schema
        except Exception as e:
            logger.warning(f"Failed to fetch from {gateway}: {e}")
            continue
    
    logger.error(f"❌ Failed to fetch schema {cid} from all gateways")
    return None

def load_all_schemas() -> Dict[str, Dict[str, Any]]:
    """Load all schemas from IPFS."""
    schemas = {}
    logger.info("🔄 Loading all schemas from IPFS...")
    
    for schema_type, cid in IPFS_SCHEMA_CIDS.items():
        logger.info(f"📋 Loading {schema_type} schema...")
        schema = fetch_schema_from_ipfs(cid)
        if schema:
            schemas[schema_type] = schema
            logger.info(f"✅ Loaded {schema_type} schema with {len(schema.get('properties', {}))} properties")
        else:
            logger.error(f"❌ Failed to load {schema_type} schema")
        time.sleep(1)  # Be nice to IPFS gateways
    
    logger.info(f"📊 Loaded {len(schemas)} schemas")
    return schemas

def get_schema_for_file(filename: str, schemas: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Determine which schema to use for a given file."""
    filename_lower = filename.lower()
    
    # File type detection - more specific conditions first
    if "lot" in filename_lower:
        schema = schemas.get("lot")
        logger.info(f"🔍 Selected lot schema for {filename} (properties: {len(schema.get('properties', {})) if schema else 0})")
        return schema
    elif "layout" in filename_lower:
        schema = schemas.get("layout")
        logger.info(f"🔍 Selected layout schema for {filename} (properties: {len(schema.get('properties', {})) if schema else 0})")
        return schema
    elif "structure" in filename_lower:
        schema = schemas.get("structure")
        logger.info(f"🔍 Selected structure schema for {filename} (properties: {len(schema.get('properties', {})) if schema else 0})")
        return schema
    elif "utility" in filename_lower:
        schema = schemas.get("utility")
        logger.info(f"🔍 Selected utility schema for {filename} (properties: {len(schema.get('properties', {})) if schema else 0})")
        return schema
    elif "appliance" in filename_lower:
        schema = schemas.get("appliance")
        logger.info(f"🔍 Selected appliance schema for {filename} (properties: {len(schema.get('properties', {})) if schema else 0})")
        return schema
    elif filename_lower == "property.json":
        schema = schemas.get("property")
        logger.info(f"🔍 Selected property schema for {filename} (properties: {len(schema.get('properties', {})) if schema else 0})")
        return schema
    elif filename_lower.startswith("file_") or "photo_metadata" in filename_lower:
        schema = schemas.get("file")
        logger.info(f"🔍 Selected file schema for {filename} (properties: {len(schema.get('properties', {})) if schema else 0})")
        return schema
    
    logger.info(f"🔍 No schema found for {filename}")
    return None

def get_enum_values(schema: Dict[str, Any], field_path: List[str]) -> Optional[List[str]]:
    """Get enum values for a field in the schema."""
    current = schema
    for part in field_path:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    
    if isinstance(current, dict) and "enum" in current:
        return current["enum"]
    return None

def get_field_type(schema: Dict[str, Any], field_path: List[str]) -> Optional[str]:
    """Get the type of a field in the schema."""
    current = schema
    for part in field_path:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    
    if isinstance(current, dict) and "type" in current:
        return current["type"]
    return None

def fix_enum_value(data: Dict[str, Any], field_path: List[str], enum_values: List[str]) -> bool:
    """Fix an enum value by setting it to null if invalid."""
    current = data
    for part in field_path[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False
    
    field_name = field_path[-1]
    if isinstance(current, dict) and field_name in current:
        current_value = current[field_name]
        if current_value not in enum_values:
            current[field_name] = None
            logger.info(f"  🔧 Fixed enum field {'.'.join(field_path)}: {current_value} -> null")
            return True
    return False

def fix_boolean_value(data: Dict[str, Any], field_path: List[str]) -> bool:
    """Fix a boolean value by converting string booleans."""
    current = data
    for part in field_path[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False
    
    field_name = field_path[-1]
    if isinstance(current, dict) and field_name in current:
        current_value = current[field_name]
        if isinstance(current_value, str):
            if current_value.lower() in ["true", "yes", "present", "available"]:
                current[field_name] = True
                logger.info(f"  🔧 Fixed boolean field {'.'.join(field_path)}: {current_value} -> true")
                return True
            elif current_value.lower() in ["false", "no", "absent", "unavailable"]:
                current[field_name] = False
                logger.info(f"  🔧 Fixed boolean field {'.'.join(field_path)}: {current_value} -> false")
                return True
    return False

def fix_string_value(data: Dict[str, Any], field_path: List[str]) -> bool:
    """Fix a string value by converting null to empty string."""
    current = data
    for part in field_path[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False
    
    field_name = field_path[-1]
    if isinstance(current, dict) and field_name in current:
        current_value = current[field_name]
        if current_value is None:
            current[field_name] = ""
            logger.info(f"  🔧 Fixed string field {'.'.join(field_path)}: null -> empty string")
            return True
    return False

def add_missing_required_fields(data: Dict[str, Any], schema: Dict[str, Any], field_path: List[str] = None, filename: str = "") -> bool:
    """Add missing required fields to the data."""
    if field_path is None:
        field_path = []
    
    modified = False
    
    # Fields that should NOT be added to structure, lot, utility, and appliance files
    file_only_fields = ["document_type", "file_format", "ipfs_url", "name", "original_url"]
    
    # Check if this is a non-file type that should exclude file-only fields
    is_non_file_type = any(keyword in filename.lower() for keyword in ["structure", "lot", "utility", "appliance"])
    
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            current_path = field_path + [prop_name]
            
            # Skip file-only fields for non-file types
            if is_non_file_type and prop_name in file_only_fields:
                logger.info(f"  ⏭️  Skipping file-only field '{prop_name}' for {filename}")
                continue
            
            # Check if field is required
            is_required = prop_name in schema.get("required", [])
            
            # Navigate to the field in data
            current_data = data
            for part in field_path:
                if isinstance(current_data, dict) and part in current_data:
                    current_data = current_data[part]
                else:
                    current_data = None
                    break
            
            if current_data is not None and isinstance(current_data, dict):
                if is_required and prop_name not in current_data:
                    # Add missing required field with appropriate default
                    if "type" in prop_schema:
                        if prop_schema["type"] == "string":
                            current_data[prop_name] = ""
                        elif prop_schema["type"] == "boolean":
                            current_data[prop_name] = False
                        elif prop_schema["type"] == "number":
                            current_data[prop_name] = 0
                        else:
                            current_data[prop_name] = None
                    else:
                        current_data[prop_name] = None
                    
                    logger.info(f"  ➕ Added missing required field {'.'.join(current_path)}")
                    modified = True
                
                # Recursively check nested objects
                if prop_name in current_data and isinstance(current_data[prop_name], dict):
                    if add_missing_required_fields(current_data[prop_name], prop_schema, current_path, filename):
                        modified = True
    
    return modified

def get_parcel_id_from_property_json(folder_path):
    """Extract parcel ID from property.json file in a folder."""
    property_json_path = os.path.join(folder_path, "property.json")
    
    if not os.path.exists(property_json_path):
        logger.info(f"⚠️  property.json not found in {folder_path}")
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
            logger.info(f"📋 Found parcel ID: {parcel_id} in {folder_path}")
            return parcel_id
        else:
            logger.info(f"⚠️  No parcel ID found in property.json in {folder_path}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error reading property.json in {folder_path}: {e}")
        return None

def validate_and_fix_data(data: Dict[str, Any], schema: Dict[str, Any], filename: str, file_path: Path) -> bool:
    """Validate and fix data according to schema with enhanced logic from fix_and_submit_local.py."""
    modified = False
    
    logger.info(f"🔍 Validating {filename} against schema...")
    
    # Get property_id from the directory containing this file
    property_dir = str(file_path.parent)
    property_id = get_parcel_id_from_property_json(property_dir)
    
    # Skip the main relationship file and CID files - never fix them
    if filename == "bafkreibzrfmqka5h7dnuz7jzilgx4ht5rqcrx3ocl23nger65frbb5hzma.json" or (filename.endswith(".json") and filename.startswith("bafkre")):
        logger.info(f"⏭️  Skipping CID file: {filename}")
        return False
    
    # Check if this is a non-file type that should exclude file-specific fields
    is_non_file_type = any(keyword in filename.lower() for keyword in ["structure", "lot", "utility", "appliance"])
    
    # Remove file-specific fields from non-file types if they exist
    if is_non_file_type:
        file_only_fields = ["document_type", "file_format", "ipfs_url", "name", "original_url"]
        for field in file_only_fields:
            if field in data:
                del data[field]
                modified = True
                logger.info(f"  🗑️  Removed file-only field '{field}' from {filename}")
    
    # Fix source_http_request (but not for relationship files)
    if not filename.startswith("relationship_"):
        # Add source_http_request if it doesn't exist
        if "source_http_request" not in data:
            data["source_http_request"] = {
                "method": "GET",
                "url": "https://pbcpao.gov/Property/Details"
            }
            modified = True
            logger.info(f"  ➕ Added source_http_request")
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
    
    # Fix request_identifier to use parcel_id if available - for ALL files (except relationship files)
    if not filename.startswith("relationship_"):
        # Use property_id if available, otherwise use "N/A"
        request_id_value = property_id if property_id else "N/A"
        
        if "request_identifier" not in data:
            data["request_identifier"] = request_id_value
            modified = True
            logger.info(f"  ➕ Added request_identifier: {request_id_value}")
        elif data["request_identifier"] != request_id_value:
            data["request_identifier"] = request_id_value
            modified = True
            logger.info(f"  🔧 Updated request_identifier to: {request_id_value}")
    
    # Ensure appliance, lot, structure, and utility files have request_identifier (double-check)
    if any(keyword in filename for keyword in ["appliance", "lot", "structure", "utility"]) and not filename.startswith("relationship_"):
        request_id_value = property_id if property_id else "N/A"
        if "request_identifier" not in data:
            data["request_identifier"] = request_id_value
            modified = True
            logger.info(f"  ➕ Added request_identifier to {filename}: {request_id_value}")
        elif data["request_identifier"] != request_id_value:
            data["request_identifier"] = request_id_value
            modified = True
            logger.info(f"  🔧 Updated request_identifier in {filename} to: {request_id_value}")
    
    # Also ensure layout files have request_identifier
    if "layout" in filename and not filename.startswith("relationship_"):
        request_id_value = property_id if property_id else "N/A"
        if "request_identifier" not in data:
            data["request_identifier"] = request_id_value
            modified = True
            logger.info(f"  ➕ Added request_identifier to layout file {filename}: {request_id_value}")
        elif data["request_identifier"] != request_id_value:
            data["request_identifier"] = request_id_value
            modified = True
            logger.info(f"  🔧 Updated request_identifier in layout file {filename} to: {request_id_value}")
    
    # Ensure file document_type and file_format for file_xxxxxx files (but not relationship files)
    if filename.startswith("file_") and "photo_metadata" in filename and not filename.startswith("relationship_"):
        if "document_type" not in data or data["document_type"] != "PropertyImage":
            data["document_type"] = "PropertyImage"
            modified = True
            logger.info(f"  🔧 Set document_type to PropertyImage")
        
        # Fix file_format to be jpeg for photo metadata files
        if "file_format" not in data or data["file_format"] != "jpeg":
            data["file_format"] = "jpeg"
            modified = True
            logger.info(f"  🔧 Set file_format to jpeg")
    
    # Ensure layout files have source_http_request (but not relationship files)
    if "layout" in filename and not filename.startswith("relationship_"):
        if "source_http_request" not in data:
            data["source_http_request"] = {
                "method": "GET",
                "url": "https://pbcpao.gov/Property/Details"
            }
            modified = True
            logger.info(f"  ➕ Added source_http_request to layout file")
    
    # Add missing required fields (excluding file-only fields for non-file types)
    if add_missing_required_fields(data, schema, filename=filename):
        modified = True
    
    # Get all boolean fields from the schema dynamically
    def get_boolean_fields_from_schema(schema_obj, path=""):
        """Recursively find all boolean fields in the schema."""
        boolean_fields = []
        
        if isinstance(schema_obj, dict):
            if schema_obj.get("type") == "boolean":
                boolean_fields.append(path)
            elif schema_obj.get("type") == "object" and "properties" in schema_obj:
                for prop_name, prop_schema in schema_obj["properties"].items():
                    new_path = f"{path}.{prop_name}" if path else prop_name
                    boolean_fields.extend(get_boolean_fields_from_schema(prop_schema, new_path))
            elif schema_obj.get("type") == "array" and "items" in schema_obj:
                boolean_fields.extend(get_boolean_fields_from_schema(schema_obj["items"], path))
        
        return boolean_fields
    
    # Get all boolean fields from the schema
    schema_boolean_fields = get_boolean_fields_from_schema(schema)
    logger.info(f"📋 Found boolean fields in schema: {schema_boolean_fields}")
    
    # Debug: Check if schema has properties
    if schema and "properties" in schema:
        logger.info(f"🔍 Schema has {len(schema['properties'])} properties")
        logger.info(f"🔍 Schema keys: {list(schema.keys())}")
        for prop_name, prop_schema in schema["properties"].items():
            if isinstance(prop_schema, dict) and prop_schema.get("type") == "boolean":
                logger.info(f"🔍 Found boolean property: {prop_name}")
    else:
        logger.warning(f"⚠️  Schema missing or has no properties: {schema.keys() if schema else 'None'}")
    
    # Fix boolean fields found in the schema
    for field_path in schema_boolean_fields:
        # Handle nested paths like "property.has_garage"
        field_parts = field_path.split('.')
        current_obj = data
        
        # Navigate to the nested object
        for part in field_parts[:-1]:
            if part in current_obj and isinstance(current_obj[part], dict):
                current_obj = current_obj[part]
            else:
                break
        else:
            # We found the object, now check the actual field
            field_name = field_parts[-1]
            if field_name in current_obj:
                value = current_obj[field_name]
                if not isinstance(value, bool):
                    if isinstance(value, str):
                        if value.lower() in ["true", "yes", "present", "available"]:
                            current_obj[field_name] = True
                            logger.info(f"  🔧 Fixed schema boolean field {field_path}: {value} -> true")
                            modified = True
                        else:
                            current_obj[field_name] = False
                            logger.info(f"  🔧 Fixed schema boolean field {field_path}: {value} -> false")
                            modified = True
                    elif value is None:
                        current_obj[field_name] = False
                        logger.info(f"  🔧 Fixed schema boolean field {field_path}: null -> false")
                        modified = True
                    else:
                        current_obj[field_name] = False
                        logger.info(f"  🔧 Fixed schema boolean field {field_path}: {value} -> false")
                        modified = True
    
    # Fix enum values and boolean fields
    def fix_enums_recursive(obj: Any, schema_obj: Dict[str, Any], path: List[str] = None):
        nonlocal modified
        if path is None:
            path = []
        
        if isinstance(obj, dict) and isinstance(schema_obj, dict):
            for key, value in obj.items():
                current_path = path + [key]
                
                if key in schema_obj:
                    field_schema = schema_obj[key]
                    
                    # Check for enum values
                    if "enum" in field_schema:
                        enum_values = field_schema["enum"]
                        if value not in enum_values:
                            obj[key] = None
                            logger.info(f"  🔧 Fixed enum field {'.'.join(current_path)}: {value} -> null")
                            modified = True
                    
                    # Check for boolean type - enhanced logic from fix_and_submit_local.py
                    elif field_schema.get("type") == "boolean":
                        if isinstance(value, str):
                            if value.lower() in ["true", "yes", "present", "available"]:
                                obj[key] = True
                                logger.info(f"  🔧 Fixed boolean field {'.'.join(current_path)}: {value} -> true")
                                modified = True
                            elif value.lower() in ["false", "no", "absent", "unavailable"]:
                                obj[key] = False
                                logger.info(f"  🔧 Fixed boolean field {'.'.join(current_path)}: {value} -> false")
                                modified = True
                            else:
                                # Any other string value, set to false
                                obj[key] = False
                                logger.info(f"  🔧 Fixed invalid boolean field {'.'.join(current_path)}: {value} -> false")
                                modified = True
                        elif value is None:
                            # Set null boolean fields to false
                            obj[key] = False
                            logger.info(f"  🔧 Fixed null boolean field {'.'.join(current_path)}: null -> false")
                            modified = True
                        elif not isinstance(value, bool):
                            # Any non-boolean value, set to false
                            obj[key] = False
                            logger.info(f"  🔧 Fixed non-boolean field {'.'.join(current_path)}: {value} -> false")
                            modified = True
                    
                    # Check for string type
                    elif field_schema.get("type") == "string":
                        if value is None:
                            obj[key] = ""
                            logger.info(f"  🔧 Fixed string field {'.'.join(current_path)}: null -> empty string")
                            modified = True
                    
                    # Recursively check nested objects
                    if isinstance(value, dict) and "properties" in field_schema:
                        fix_enums_recursive(value, field_schema.get("properties", {}), current_path)
    
    # Start recursive validation
    fix_enums_recursive(data, schema.get("properties", {}))
    
    return modified

def process_file(file_path: Path, schemas: Dict[str, Dict[str, Any]]) -> bool:
    """Process a single file and fix schema issues."""
    try:
        # Skip relationship files
        if "relationship" in file_path.name:
            logger.info(f"⏭️  Skipping relationship file: {file_path.name}")
            return False
        
        # Load the file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Get appropriate schema
        schema = get_schema_for_file(file_path.name, schemas)
        if not schema:
            logger.info(f"⏭️  No schema found for {file_path.name}")
            return False
        
        # Validate and fix
        modified = validate_and_fix_data(data, schema, file_path.name, file_path)
        
        if modified:
            # Save the fixed data
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"✅ Fixed and saved: {file_path.name}")
            return True
        else:
            logger.info(f"✅ No fixes needed: {file_path.name}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error processing {file_path}: {e}")
        return False

def process_directory(directory_path: str) -> None:
    """Process all JSON files in a directory and its subdirectories."""
    directory = Path(directory_path)
    if not directory.exists():
        logger.error(f"❌ Directory not found: {directory_path}")
        return
    
    logger.info(f"🚀 Starting schema validation fix for: {directory_path}")
    
    # Load all schemas
    schemas = load_all_schemas()
    if not schemas:
        logger.error("❌ No schemas loaded. Cannot proceed.")
        return
    
    # Find all JSON files
    json_files = list(directory.rglob("*.json"))
    logger.info(f"📁 Found {len(json_files)} JSON files to process")
    
    # Process files
    fixed_count = 0
    for file_path in json_files:
        if process_file(file_path, schemas):
            fixed_count += 1
    
    logger.info(f"🎉 Processing complete!")
    logger.info(f"📊 Fixed {fixed_count} out of {len(json_files)} files")

def main():
    """Main function."""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Process output directory
    output_dir = "output"
    if os.path.exists(output_dir):
        process_directory(output_dir)
    else:
        logger.error(f"❌ Output directory not found: {output_dir}")
        logger.info("💡 You can also specify a different directory:")
        logger.info("   python fix_schema_validation.py <directory_path>")

if __name__ == "__main__":
    main() 