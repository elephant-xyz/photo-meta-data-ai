#!/usr/bin/env python3
"""
AI Image Analysis Script - Modified to remove layout to file relationships
"""

import os
import sys
import json
import base64
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datetime import datetime
import requests
from math import ceil
from PIL import Image
import io
import uuid
import threading
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import tempfile
import logging
from dotenv import load_dotenv

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/ai-analyzer.log')
            # Removed StreamHandler to only log to files
        ]
    )
    return logging.getLogger(__name__)

# Load environment variables from .env file
def load_environment():
    """Load environment variables from .env file"""
    env_paths = [
        '.env',
        '/content/.env',
        os.path.expanduser('~/.env')
    ]
    
    env_loaded = False
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            # Environment loaded successfully (no console output)
            env_loaded = True
            break
    
    if not env_loaded:
        # No .env file found (no console output)
        pass
    
    return env_loaded

# Load environment variables
load_environment()

logger = setup_logging()

# --- CONFIG ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("❌ OPENAI_API_KEY not found in environment variables")
    sys.exit(1)
else:
    logger.info(f"✓ OpenAI API key loaded: {openai_api_key[:10]}...")

client = OpenAI(api_key=openai_api_key)

# S3 Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai')
S3_BASE_PREFIX = os.getenv('S3_BASE_PREFIX', '')  # Will be set per property

# IPFS Schema CIDs
IPFS_SCHEMA_CIDS = {
    "lot": "bafkreigy3tsgcwtgz4nu5jc7cnkb6bizpbxbn3rh6ectz44z6f3tqfjdum",
    "layout": "bafkreihrwupbrldaxwm2qcuryrug5pwplxmp4ckdo7whqz52zwuw5j7l2q",
    "structure": "bafkreid2wa56cecrm6ge4slmsm56xqy6j3gqlhldrljmruh64ams542xxe",
    "utility": "bafkreihuoaw6fm5abblivzgepkxkhduc5mivhho4rcidp5lvgb7fhyuide",
    "appliance": "bafkreieew4njulmeecnm3kah7w43eiali6lre5o45ttiyaqfjhb3ecu2mq",
    "file": "bafkreihug7qtvmblmpgdox7ex476inddz4u365gl33epmqoatiecqjveqq",
    "property": "bafkreih6x76aedhs7lqjk5uq4zskmfs33agku62b4flpq5s5pa6aek2gga"

}

# Relationship schema CID
RELATIONSHIP_SCHEMA_CID = "bafkreicaq62gggwbppihgstao2maakafmghjttf73ai53tz5tam2cixrvu"

SCHEMA_FOLDER = "schema"
OUTPUT_BASE_FOLDER = "output"
SCHEMA_KEYS = [
    "property", "lot",
    "layout", "structure", "utility", "appliance"
]

# Image optimization settings
MAX_IMAGE_SIZE = (1024, 1024)
JPEG_QUALITY = 85

# Parallel processing settings
MAX_CONCURRENT_REQUESTS = 5  # Number of parallel API requests (be careful with rate limits)
MAX_WORKERS = 10  # Number of threads for overall processing

# Global tracking variables (thread-safe with locks)
stats_lock = threading.Lock()
TOTAL_IMAGES_PROCESSED = 0
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
TOTAL_COST = 0.0

# Initialize S3 client
s3_client = None

def fetch_schema_from_ipfs(cid):
    """Fetch schema from IPFS CID"""
    try:
        response = requests.get(f"https://ipfs.io/ipfs/{cid}", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch schema from IPFS: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching schema from IPFS: {e}")
        return None

def authenticate_aws():
    """Authenticate with AWS"""
    global s3_client
    
    try:
        s3_client = boto3.client('s3')
        # Test the connection
        s3_client.list_buckets()
        logger.info("✅ AWS S3 authentication successful")
        return True
    except NoCredentialsError:
        logger.error("❌ AWS credentials not found")
        return False
    except Exception as e:
        logger.error(f"❌ AWS authentication failed: {e}")
        return False

def list_s3_folders():
    """List all folders in S3 bucket"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Delimiter='/'
        )
        
        folders = []
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                folder_name = prefix['Prefix'].rstrip('/')
                folders.append(folder_name)
        
        return folders
    except Exception as e:
        logger.error(f"Error listing S3 folders: {e}")
        return []

def list_s3_property_folders():
    """List property folders in S3 bucket"""
    folders = list_s3_folders()
    property_folders = [f for f in folders if f.isdigit()]
    return property_folders

def list_s3_subfolders():
    """List all subfolders in S3 bucket"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Delimiter='/'
        )
        
        subfolders = []
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                subfolder_name = prefix['Prefix'].rstrip('/')
                subfolders.append(subfolder_name)
        
        return subfolders
    except Exception as e:
        logger.error(f"Error listing S3 subfolders: {e}")
        return []

def list_s3_subfolders_for_property(property_id):
    """List subfolders for a specific property"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=f"{property_id}/",
            Delimiter='/'
        )
        
        subfolders = []
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                subfolder_name = prefix['Prefix'].rstrip('/')
                # Extract just the subfolder name (remove property_id/)
                if subfolder_name.startswith(f"{property_id}/"):
                    subfolder_name = subfolder_name[len(f"{property_id}/"):]
                subfolders.append(subfolder_name)
        
        return subfolders
    except Exception as e:
        logger.error(f"Error listing subfolders for property {property_id}: {e}")
        return []

def list_s3_images_in_folder(folder_name, property_id=None):
    """List all images in a specific S3 folder"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=f"{folder_name}/"
        )
        
        image_objects = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                # Check if it's an image file
                if any(key.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']):
                    image_objects.append({
                        'key': key,
                        'name': os.path.basename(key)
                    })
        
        return image_objects
    except Exception as e:
        logger.error(f"Error listing images in folder {folder_name}: {e}")
        return []

def list_s3_images_in_property(folder_name):
    """List all images in entire property folder (including all subfolders)"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=f"{folder_name}/"
        )
        
        image_objects = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                # Check if it's an image file
                if any(key.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']):
                    image_objects.append({
                        'key': key,
                        'name': os.path.basename(key)
                    })
        
        return image_objects
    except Exception as e:
        logger.error(f"Error listing images in property {folder_name}: {e}")
        return []

def download_s3_image_to_temp(s3_key):
    """Download S3 image to temporary file"""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        image_data = response['Body'].read()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file.write(image_data)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        logger.error(f"Error downloading S3 image {s3_key}: {e}")
        return None

def get_next_batch_number(output_dir):
    """Get the next batch number based on existing files"""
    try:
        if not os.path.exists(output_dir):
            return 1
        
        existing_files = os.listdir(output_dir)
        batch_numbers = []
        
        for filename in existing_files:
            if filename.startswith('batch_') and filename.endswith('.json'):
                try:
                    batch_num = int(filename.split('_')[1].split('.')[0])
                    batch_numbers.append(batch_num)
                except:
                    continue
        
        if not batch_numbers:
            return 1
        
        return max(batch_numbers) + 1
    except Exception as e:
        logger.error(f"Error getting next batch number: {e}")
        return 1

def generate_placeholder_cid(prefix, identifier):
    """Generate a placeholder CID for testing"""
    return f"{prefix}_{identifier}"

def create_relationship(from_cid, to_cid, relationship_type, relationship_schema=None):
    """Create a relationship object"""
    relationship = {
        "from": from_cid,
        "to": to_cid,
        "type": relationship_type
    }
    
    if relationship_schema:
        relationship["schema"] = relationship_schema
    
    return relationship

def generate_smart_relationships_from_batch(batch_data, image_paths, property_cid):
    """Generate relationships from batch data"""
    relationships = []
    
    # Property -> Image relationships
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        image_cid = generate_placeholder_cid("image", image_name)
        relationships.append(create_relationship(property_cid, image_cid, "property_has_file"))
    
    # Property -> Object relationships
    object_types = ["structure", "lot", "utility", "layout", "appliance"]
    for obj_type in object_types:
        if batch_data.get(obj_type):
            obj_cid = generate_placeholder_cid(obj_type, f"batch_{obj_type}")
            rel_type = f"property_has_{obj_type}"
            relationships.append(create_relationship(property_cid, obj_cid, rel_type))
    
    # Layout -> Image relationships (REMOVED - as requested)
    # This section has been removed to eliminate layout to file relationships
    
    return relationships

def generate_relationships_per_image(extracted_data, image_name, property_cid, image_index):
    """Generate relationships for a single image"""
    relationships = []
    
    # Property -> Image relationship
    image_cid = generate_placeholder_cid("image", image_name)
    relationships.append(create_relationship(property_cid, image_cid, "property_has_file"))
    
    # Property -> Object relationships
    object_types = ["structure", "lot", "utility", "layout", "appliance"]
    for obj_type in object_types:
        if extracted_data.get(obj_type):
            obj_cid = generate_placeholder_cid(obj_type, f"image_{image_index}_{obj_type}")
            rel_type = f"property_has_{obj_type}"
            relationships.append(create_relationship(property_cid, obj_cid, rel_type))
    
    # Layout -> Image relationships (REMOVED - as requested)
    # This section has been removed to eliminate layout to file relationships
    
    return relationships

def chunk_list(lst, chunk_size):
    """Split a list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_openai_cost_for_today(api_key=None):
    """Get OpenAI cost for today (placeholder)"""
    return 0.0

def optimize_image(image_path):
    """Optimize image for API"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large
            if img.size[0] > MAX_IMAGE_SIZE[0] or img.size[1] > MAX_IMAGE_SIZE[1]:
                img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Save optimized image
            output_path = f"{image_path}_optimized.jpg"
            img.save(output_path, 'JPEG', quality=JPEG_QUALITY, optimize=True)
            
            return output_path
    except Exception as e:
        logger.error(f"Error optimizing image {image_path}: {e}")
        return image_path

def encode_image_original(image_path):
    """Encode image to base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def optimize_s3_image(s3_key):
    """Optimize S3 image for API"""
    try:
        # Download image to temp file
        temp_path = download_s3_image_to_temp(s3_key)
        if not temp_path:
            return None
        
        # Optimize the image
        optimized_path = optimize_image(temp_path)
        
        # Encode optimized image
        encoded_image = encode_image_original(optimized_path)
        
        # Clean up temp files
        try:
            os.unlink(temp_path)
            if optimized_path != temp_path:
                os.unlink(optimized_path)
        except:
            pass
        
        return encoded_image
    except Exception as e:
        logger.error(f"Error optimizing S3 image {s3_key}: {e}")
        return None

def load_schemas_from_ipfs():
    """Load all schemas from IPFS"""
    schemas = {}
    
    for schema_name, cid in IPFS_SCHEMA_CIDS.items():
        schema_data = fetch_schema_from_ipfs(cid)
        if schema_data:
            schemas[schema_name] = schema_data
            logger.info(f"✅ Loaded {schema_name} schema from IPFS")
        else:
            logger.warning(f"⚠️ Failed to load {schema_name} schema from IPFS")
    
    # Load relationship schema
    relationship_schema = fetch_schema_from_ipfs(RELATIONSHIP_SCHEMA_CID)
    if relationship_schema:
        schemas["relationship"] = relationship_schema
        logger.info(f"✅ Loaded relationship schema from IPFS")
    else:
        logger.warning(f"⚠️ Failed to load relationship schema from IPFS")
    
    return schemas

def load_optimized_json_schema_prompt(folder_name=None, schemas=None):
    """Load optimized prompt with JSON schema"""
    base_prompt = """Analyze these real estate images and extract detailed property information. Return comprehensive JSON data about all visible property features.

Focus on:
- Property structure and layout
- Appliances and fixtures
- Lot and exterior features
- Utility systems
- Room types and spaces

Return the data in this exact JSON structure:"""

    if schemas:
        # Add schema information to prompt
        schema_info = "\n\nUse these IPFS schemas for data structure:\n"
        for schema_name, schema_data in schemas.items():
            if schema_name != "relationship":
                schema_info += f"- {schema_name}: {schema_data.get('title', 'No title')}\n"
        
        base_prompt += schema_info

    if folder_name:
        base_prompt += f"\n\nFocus on: {folder_name.replace('_', ' ').title()}"

    return base_prompt

def try_parse_json(text):
    """Try to parse JSON from text"""
    try:
        # Find JSON content between ```json and ``` markers
        if "```json" in text and "```" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            json_text = text[start:end].strip()
        else:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != 0:
                json_text = text[start:end]
            else:
                return None
        
        return json.loads(json_text)
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return None

def call_openai_optimized_s3(image_objects, prompt, schemas=None):
    """Optimized OpenAI call for S3 images with retry logic and better error handling."""
    global TOTAL_IMAGES_PROCESSED, TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, TOTAL_COST

    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            images_in_batch = len(image_objects)

            # Thread-safe update of global stats
            with stats_lock:
                TOTAL_IMAGES_PROCESSED += images_in_batch

            messages = [
                {"role": "system", "content": "You are a detailed real estate image analyzer. Your job is to carefully examine each image and provide specific, detailed analysis of visible property features. Be thorough and descriptive in your analysis. Return comprehensive JSON data about all visible property features."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]

            for i, image_obj in enumerate(image_objects):
                # Handle both local files and S3 files
                if image_obj['key'].startswith('s3://') or os.path.exists(image_obj['key']):
                    # Local file path
                    image_b64 = encode_image_original(image_obj['key'])
                else:
                    # S3 key
                    image_b64 = optimize_s3_image(image_obj['key'])
                
                if image_b64:
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    })

            # Calculate dynamic timeout based on batch size
            batch_size = len(image_objects)
            dynamic_timeout = min(180, 30 + (batch_size * 15))

            # Make API call with timeout
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=4000,
                temperature=0.1,
                timeout=dynamic_timeout
            )

            # Extract response content
            response_content = response.choices[0].message.content

            # Parse JSON from response
            result = try_parse_json(response_content)
            if not result:
                logger.warning(f"Failed to parse JSON from OpenAI response")
                continue

            # Calculate cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = (prompt_tokens * 0.01 / 1000) + (completion_tokens * 0.03 / 1000)

            # Thread-safe update of global stats
            with stats_lock:
                TOTAL_PROMPT_TOKENS += prompt_tokens
                TOTAL_COMPLETION_TOKENS += completion_tokens
                TOTAL_COST += cost

            logger.info(f"✅ OpenAI API call successful (attempt {attempt + 1})")
            logger.info(f"📊 Tokens: {prompt_tokens} prompt, {completion_tokens} completion")
            logger.info(f"💰 Cost: ${cost:.4f}")

            return result, cost

        except Exception as e:
            logger.error(f"❌ OpenAI API call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"❌ All retry attempts failed")
                return None, 0.0

    return None, 0.0

def call_openai_optimized(image_paths, prompt):
    """Optimized OpenAI call for local images with retry logic and better error handling."""
    global TOTAL_IMAGES_PROCESSED, TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, TOTAL_COST

    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            images_in_batch = len(image_paths)

            # Thread-safe update of global stats
            with stats_lock:
                TOTAL_IMAGES_PROCESSED += images_in_batch

            messages = [
                {"role": "system", "content": "You are a detailed real estate image analyzer. Your job is to carefully examine each image and provide specific, detailed analysis of visible property features. Be thorough and descriptive in your analysis. Return comprehensive JSON data about all visible property features."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]

            for image_path in image_paths:
                image_b64 = encode_image_original(image_path)
                if image_b64:
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    })

            # Calculate dynamic timeout based on batch size
            batch_size = len(image_paths)
            dynamic_timeout = min(180, 30 + (batch_size * 15))

            # Make API call with timeout
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=4000,
                temperature=0.1,
                timeout=dynamic_timeout
            )

            # Extract response content
            response_content = response.choices[0].message.content

            # Parse JSON from response
            result = try_parse_json(response_content)
            if not result:
                logger.warning(f"Failed to parse JSON from OpenAI response")
                continue

            # Calculate cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = (prompt_tokens * 0.01 / 1000) + (completion_tokens * 0.03 / 1000)

            # Thread-safe update of global stats
            with stats_lock:
                TOTAL_PROMPT_TOKENS += prompt_tokens
                TOTAL_COMPLETION_TOKENS += completion_tokens
                TOTAL_COST += cost

            logger.info(f"✅ OpenAI API call successful (attempt {attempt + 1})")
            logger.info(f"📊 Tokens: {prompt_tokens} prompt, {completion_tokens} completion")
            logger.info(f"💰 Cost: ${cost:.4f}")

            return result, cost

        except Exception as e:
            logger.error(f"❌ OpenAI API call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"❌ All retry attempts failed")
                return None, 0.0

    return None, 0.0

def generate_image_json_files_s3(image_objects, output_dir, batch_number):
    """Generate individual JSON files for S3 images"""
    image_files = {}
    
    for i, image_obj in enumerate(image_objects):
        image_name = image_obj['name']
        image_cid = image_name.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        
        # Create image data structure
        image_data = {
            "filename": image_name,
            "s3_key": image_obj['key'],
            "batch_number": batch_number,
            "image_index": i
        }
        
        # Save to file
        filename = f"{image_cid}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(image_data, f, indent=2)
        
        image_files[image_cid] = filename
        logger.info(f"✅ Generated image file: {filename}")
    
    return image_files

def generate_image_json_files(image_paths, output_dir, batch_number):
    """Generate individual JSON files for local images"""
    image_files = {}
    
    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        image_cid = image_name.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        
        # Create image data structure
        image_data = {
            "filename": image_name,
            "local_path": image_path,
            "batch_number": batch_number,
            "image_index": i
        }
        
        # Save to file
        filename = f"{image_cid}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(image_data, f, indent=2)
        
        image_files[image_cid] = filename
        logger.info(f"✅ Generated image file: {filename}")
    
    return image_files

def generate_individual_object_files_s3(batch_data, image_objects, output_dir, batch_number):
    """Generate individual object files for S3 images"""
    object_files = {
        "layouts": {},
        "appliances": {},
        "property_objects": {},
        "images": {}
    }
    
    # Generate image files
    image_files = generate_image_json_files_s3(image_objects, output_dir, batch_number)
    object_files["images"] = image_files
    
    # Generate layout files
    if batch_data.get("layouts"):
        for layout_key, layout_data in batch_data["layouts"].items():
            filename = f"layout_{layout_key}_{batch_number}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(layout_data, f, indent=2)
            
            object_files["layouts"][layout_key] = filename
            logger.info(f"✅ Generated layout file: {filename}")
    
    # Generate appliance files
    if batch_data.get("appliances"):
        for appliance_key, appliance_data in batch_data["appliances"].items():
            filename = f"appliance_{appliance_key}_{batch_number}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(appliance_data, f, indent=2)
            
            object_files["appliances"][appliance_key] = filename
            logger.info(f"✅ Generated appliance file: {filename}")
    
    # Generate property object files
    property_objects = ["structure", "lot", "utility"]
    for obj_type in property_objects:
        if batch_data.get(obj_type):
            filename = f"{obj_type}_{batch_number}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(batch_data[obj_type], f, indent=2)
            
            object_files["property_objects"][obj_type] = filename
            logger.info(f"✅ Generated {obj_type} file: {filename}")
    
    return object_files

def generate_individual_object_files(batch_data, image_paths, output_dir, batch_number):
    """Generate individual object files for local images"""
    object_files = {
        "layouts": {},
        "appliances": {},
        "property_objects": {},
        "images": {}
    }
    
    # Generate image files
    image_files = generate_image_json_files(image_paths, output_dir, batch_number)
    object_files["images"] = image_files
    
    # Generate layout files
    if batch_data.get("layouts"):
        for layout_key, layout_data in batch_data["layouts"].items():
            filename = f"layout_{layout_key}_{batch_number}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(layout_data, f, indent=2)
            
            object_files["layouts"][layout_key] = filename
            logger.info(f"✅ Generated layout file: {filename}")
    
    # Generate appliance files
    if batch_data.get("appliances"):
        for appliance_key, appliance_data in batch_data["appliances"].items():
            filename = f"appliance_{appliance_key}_{batch_number}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(appliance_data, f, indent=2)
            
            object_files["appliances"][appliance_key] = filename
            logger.info(f"✅ Generated appliance file: {filename}")
    
    # Generate property object files
    property_objects = ["structure", "lot", "utility"]
    for obj_type in property_objects:
        if batch_data.get(obj_type):
            filename = f"{obj_type}_{batch_number}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(batch_data[obj_type], f, indent=2)
            
            object_files["property_objects"][obj_type] = filename
            logger.info(f"✅ Generated {obj_type} file: {filename}")
    
    return object_files

def generate_clean_cid(object_type, identifier):
    """Generate a clean CID for an object"""
    return f"{object_type}_{identifier}"

def generate_relationships_from_object_files_s3(object_files, image_objects, property_cid, batch_number, property_id, relationship_schema=None):
    """
    Generate relationships using clean, meaningful CIDs based on filenames.
    MODIFIED: Removed layout to file relationships as requested.
    """
    relationships = []

    # Get image CIDs from the generated image files (these are already clean filenames)
    image_cids = object_files.get("images", {})  # This is a dict: {image_cid -> filename}
    
    if not image_cids:
        logger.warning(f"No image files found in object_files. Available keys: {list(object_files.keys())}")
        return relationships

    logger.info(f"Found {len(image_cids)} image CIDs for relationship generation")

    # Create property -> image relationships using image CIDs (which are filenames without .json)
    for image_cid in image_cids.keys():
        relationships.append(create_relationship(property_cid, image_cid, "property_has_file", relationship_schema))

    # Property-level object relationships using filenames as CIDs
    for obj_type, filename in object_files["property_objects"].items():
        # Use filename without .json extension as CID
        obj_cid = filename.replace(".json", "")

        # Map object types to relationship types
        relationship_mapping = {
            "structure": "property_has_structure",
            "lot": "property_has_lot",
            "utility": "property_has_utilities"
        }

        rel_type = relationship_mapping.get(obj_type, f"property_has_{obj_type}")
        relationships.append(create_relationship(property_cid, obj_cid, rel_type, relationship_schema))

    # Layout relationships using filenames as CIDs
    # MODIFIED: Removed layout to file relationships as requested
    for layout_key, filename in object_files["layouts"].items():
        # Use filename without .json extension as CID
        layout_cid = filename.replace(".json", "")
        relationships.append(create_relationship(property_cid, layout_cid, "property_has_layout", relationship_schema))
        
        # REMOVED: Layout to image relationships
        # for image_cid in image_cids.keys():
        #     relationships.append(create_relationship(layout_cid, image_cid, "layout_has_image", relationship_schema))

    # Appliance relationships using filenames as CIDs
    for appliance_key, filename in object_files["appliances"].items():
        # Use filename without .json extension as CID
        appliance_cid = filename.replace(".json", "")
        relationships.append(create_relationship(property_cid, appliance_cid, "property_has_appliance", relationship_schema))

    return relationships

def generate_relationships_from_object_files(object_files, image_paths, property_cid, batch_number, folder_path):
    """
    Generate relationships using clean, meaningful CIDs based on filenames.
    MODIFIED: Removed layout to file relationships as requested.
    """
    relationships = []

    # Get image CIDs from the generated image files (these are already clean filenames)
    image_cids = object_files.get("images", {})  # This is a dict: {image_cid -> filename}
    
    if not image_cids:
        logger.warning(f"No image files found in object_files. Available keys: {list(object_files.keys())}")
        return relationships

    logger.info(f"Found {len(image_cids)} image CIDs for relationship generation")

    # Create property -> image relationships using image CIDs (which are filenames without .json)
    for image_cid in image_cids.keys():
        relationships.append(create_relationship(property_cid, image_cid, "property_has_file"))

    # Property-level object relationships using filenames as CIDs
    for obj_type, filename in object_files["property_objects"].items():
        # Use filename without .json extension as CID
        obj_cid = filename.replace(".json", "")

        # Map object types to relationship types
        relationship_mapping = {
            "structure": "property_has_structure",
            "lot": "property_has_lot",
            "utility": "property_has_utilities"
        }

        rel_type = relationship_mapping.get(obj_type, f"property_has_{obj_type}")
        relationships.append(create_relationship(property_cid, obj_cid, rel_type))

    # Layout relationships using filenames as CIDs
    # MODIFIED: Removed layout to file relationships as requested
    for layout_key, filename in object_files["layouts"].items():
        # Use filename without .json extension as CID
        layout_cid = filename.replace(".json", "")
        relationships.append(create_relationship(property_cid, layout_cid, "property_has_layout"))
        
        # REMOVED: Layout to image relationships
        # for image_cid in image_cids.keys():
        #     relationships.append(create_relationship(layout_cid, image_cid, "layout_has_image"))

    # Appliance relationships using filenames as CIDs
    for appliance_key, filename in object_files["appliances"].items():
        # Use filename without .json extension as CID
        appliance_cid = filename.replace(".json", "")
        relationships.append(create_relationship(property_cid, appliance_cid, "property_has_appliance"))

    return relationships

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Image Analysis Script - No Layout to File Relationships')
    parser.add_argument('--property-id', type=str, help='Specific property ID to process')
    parser.add_argument('--all-properties', action='store_true', help='Process all properties')
    parser.add_argument('--local-folders', action='store_true', help='Process from local folders')
    parser.add_argument('--help', action='store_true', help='Show this help message')
    
    args = parser.parse_args()
    
    print("🚀 AI Image Analysis Script - Modified (No Layout to File Relationships)")
    print("=" * 70)
    print("This version has been modified to remove layout to file relationships")
    print("as requested by the user.")
    print("=" * 70)
    
    if args.help:
        parser.print_help()
        return
    
    print("✅ Script loaded successfully!")
    print("📝 Use --help for available options")

if __name__ == "__main__":
    main() 