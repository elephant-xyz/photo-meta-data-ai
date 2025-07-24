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
try:
    from .image_converter import ImageConverter, convert_to_webp_base64
except ImportError:
    from image_converter import ImageConverter, convert_to_webp_base64

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

# County data directory
COUNTY_DATA_DIR = "county-data"




SCHEMA_FOLDER = "schema"
OUTPUT_BASE_FOLDER = "output"
SCHEMA_KEYS = [
    "property", "lot",
    "layout", "structure", "utility", "appliance"
]
# VISUAL_TAG_FILE = "./schema/visual-tags.pdf"  # Removed dependency on PDF file

# Image optimization settings
MAX_IMAGE_SIZE = (800, 800)  # Reduced for faster Colab processing
JPEG_QUALITY = 75  # Reduced for faster uploads

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


def load_county_layout_data(property_id):
    """Load layout data from county-data directory for a specific property."""
    county_layouts = []

    if not os.path.exists(COUNTY_DATA_DIR):
        logger.warning(f"⚠️  County data directory not found: {COUNTY_DATA_DIR}")
        return county_layouts

    property_dir = os.path.join(COUNTY_DATA_DIR, str(property_id))
    if not os.path.exists(property_dir):
        logger.warning(f"⚠️  Property directory not found: {property_dir}")
        return county_layouts

    # Find all layout files
    layout_files = []
    for file in os.listdir(property_dir):
        if file.startswith("layout_") and file.endswith(".json"):
            layout_files.append(file)

    # Sort by layout number
    layout_files.sort(key=lambda x: int(x.replace("layout_", "").replace(".json", "")))

    logger.info(f"📁 Found {len(layout_files)} layout files for property {property_id}")

    # Load each layout file
    for layout_file in layout_files:
        try:
            file_path = os.path.join(property_dir, layout_file)
            with open(file_path, 'r') as f:
                layout_data = json.load(f)

            county_layouts.append(layout_data)
            logger.info(f"📋 Loaded {layout_file}: {layout_data.get('space_type', 'unknown')}")

        except Exception as e:
            logger.error(f"❌ Error loading {layout_file}: {e}")

    return county_layouts



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
            logger.info(f"Trying to fetch {cid} from {gateway}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            schema_data = response.json()
            logger.info(f"✓ Successfully fetched schema from {gateway}")
            return schema_data
        except Exception as e:
            logger.warning(f"Error fetching from {gateway}: {e}")
            continue

    logger.error(f"Failed to fetch schema from IPFS CID {cid} from all gateways")
    return None

def authenticate_aws():
    """Authenticate with AWS S3 using environment variables"""
    global s3_client
    try:
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

        if not aws_access_key or not aws_secret_key:
            logger.error("Error: AWS credentials not found in environment variables!")
            logger.error("Please set the following environment variables:")
            logger.error("- AWS_ACCESS_KEY_ID")
            logger.error("- AWS_SECRET_ACCESS_KEY")
            logger.error("- AWS_DEFAULT_REGION (optional, defaults to us-east-1)")
            return False

        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

        # Test connection
        s3_client.list_buckets()
        logger.info("✓ AWS S3 authentication successful")
        return True

    except NoCredentialsError:
        logger.error("Error: AWS credentials not found!")
        return False
    except ClientError as e:
        logger.error(f"Error: AWS authentication failed - {e}")
        return False

def list_s3_folders():
    """List all subfolders in the S3 bucket under the base prefix"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=S3_BASE_PREFIX,
            Delimiter='/'
        )

        folders = []
        for prefix in response.get('CommonPrefixes', []):
            folder_name = prefix['Prefix'].rstrip('/').split('/')[-1]
            folders.append(folder_name)

        return folders
    except Exception as e:
        logger.error(f"Error listing S3 folders: {e}")
        return []

def list_s3_property_folders():
    """List property ID folders (not subfolders) in the S3 bucket"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Delimiter='/'
        )

        properties = []
        for prefix in response.get('CommonPrefixes', []):
            property_id = prefix['Prefix'].rstrip('/')
            properties.append(property_id)
        return properties
    except Exception as e:
        logger.error(f"Error listing S3 property folders: {e}")
        return []

def list_s3_subfolders():
    """List all subfolders (kitchen, bedroom, bathroom, etc.) under the property ID"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=S3_BASE_PREFIX,
            Delimiter='/'
        )

        subfolders = []
        for prefix in response.get('CommonPrefixes', []):
            folder_name = prefix['Prefix'].rstrip('/').split('/')[-1]
            subfolders.append(folder_name)
        return subfolders
    except Exception as e:
        logger.error(f"Error listing S3 subfolders: {e}")
        return []

def list_s3_subfolders_for_property(property_id):
    """List all category folders for a specific property in S3."""
    try:
        property_prefix = f"{property_id}/"

        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=property_prefix,
            Delimiter='/'
        )

        categories = []
        for prefix in response.get('CommonPrefixes', []):
            category_name = prefix['Prefix'].rstrip('/').split('/')[-1]
            categories.append(category_name)
        return categories
    except Exception as e:
        logger.error(f"Error listing S3 category folders for property {property_id}: {e}")
        return []

def list_s3_images_in_folder(folder_name, property_id=None):
    """List all images in a specific S3 folder"""
    try:
        if property_id:
            prefix = f"{property_id}/{folder_name}/"
        else:
            prefix = f"{S3_BASE_PREFIX}{folder_name}/"

        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=prefix
        )

        images = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append({
                    'key': key,
                    'name': os.path.basename(key),
                    'folder': folder_name
                })

        return images
    except Exception as e:
        logger.error(f"Error listing images in folder {folder_name}: {e}")
        return []

def list_s3_images_in_property(folder_name):
    """List all images in the entire property folder (including all subfolders)"""
    try:
        prefix = f"{S3_BASE_PREFIX}{folder_name}/"
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=prefix
        )

        images = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Extract subfolder name from the key
                subfolder = key.replace(prefix, '').split('/')[0] if '/' in key.replace(prefix, '') else ''
                images.append({
                    'key': key,
                    'name': os.path.basename(key),
                    'folder': folder_name,
                    'subfolder': subfolder
                })

        logger.debug(f"    [DEBUG] Found {len(images)} images in property {folder_name}")
        if images:
            subfolders = set(img['subfolder'] for img in images if img['subfolder'])
            logger.debug(f"    [DEBUG] Subfolders found: {', '.join(subfolders)}")

        return images
    except Exception as e:
        logger.error(f"Error listing images in property {folder_name}: {e}")
        return []


def download_s3_image_to_temp(s3_key):
    """Download an S3 image to a temporary file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            s3_client.download_file(S3_BUCKET_NAME, s3_key, tmp_file.name)

            # Verify file was downloaded and has content
            if os.path.exists(tmp_file.name) and os.path.getsize(tmp_file.name) > 0:
                return tmp_file.name
            else:
                logger.error(f"[ERROR] Downloaded file is empty or missing: {tmp_file.name}")
                return None
    except Exception as e:
        logger.error(f"Error downloading {s3_key}: {e}")
        return None


def get_next_batch_number(output_dir):
    """
    Find the next available batch number to avoid overwriting existing files.
    Returns the next batch number to use.
    """
    if not os.path.exists(output_dir):
        return 1

    existing_batches = []
    for filename in os.listdir(output_dir):
        if filename.startswith("batch_") and filename.endswith(".json"):
            try:
                # Extract batch number from filename like "batch_01.json"
                batch_num = int(filename.split("_")[1].split(".")[0])
                existing_batches.append(batch_num)
            except (ValueError, IndexError):
                continue

    if not existing_batches:
        return 1

    return max(existing_batches) + 1


def generate_placeholder_cid(prefix, identifier):
    """Generate a placeholder CID based on prefix and identifier."""
    return f"{prefix}_{identifier}_{uuid.uuid4().hex[:8]}"


def create_relationship(from_cid, to_cid, relationship_type, relationship_schema=None):
    """Create a relationship object following the IPFS schema or default format."""
    # For individual relationship files, use simple format with direct fields
    return {
        "from": from_cid,
        "to": to_cid,
        "type": relationship_type
    }


def generate_smart_relationships_from_batch(batch_data, image_paths, property_cid):
    """
    Generate relationships efficiently for a batch, avoiding duplicates and minimizing processing.
    This approach balances cost efficiency with relationship accuracy.
    """
    relationships = []

    # Create image CIDs for all images in the batch
    image_cids = {}
    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        image_cid = generate_placeholder_cid("image", f"{image_name.replace('.', '_')}")
        image_cids[i] = {"path": image_path, "name": image_name, "cid": image_cid}
        # Link each image to the property
        relationships.append(create_relationship(property_cid, image_cid, "property_has_file"))

    # Property-level relationships (create once per property, not per image)
    # These are shared across all images in the batch
    property_level_objects = {}

    if batch_data.get("structure"):
        structure_cid = generate_placeholder_cid("structure", f"{property_cid}_structure")
        property_level_objects["structure"] = structure_cid
        relationships.append(create_relationship(property_cid, structure_cid, "property_has_structure"))

    if batch_data.get("lot"):
        lot_cid = generate_placeholder_cid("lot", f"{property_cid}_lot")
        property_level_objects["lot"] = lot_cid
        relationships.append(create_relationship(property_cid, lot_cid, "property_has_lot"))

    if batch_data.get("utility"):
        utility_cid = generate_placeholder_cid("utility", f"{property_cid}_utility")
        property_level_objects["utility"] = utility_cid
        relationships.append(create_relationship(property_cid, utility_cid, "property_has_utilities"))

    if batch_data.get("nearby_location"):
        nearby_cid = generate_placeholder_cid("nearby", f"{property_cid}_nearby")
        property_level_objects["nearby"] = nearby_cid
        relationships.append(create_relationship(property_cid, nearby_cid, "property_has_nearby_locations"))

    # Smart layout handling - create unique layouts and link to relevant images
    layouts = batch_data.get("layout", [])
    if isinstance(layouts, dict):
        layouts = [layouts]
    elif not isinstance(layouts, list):
        layouts = []

    # Create unique layouts (avoid duplicates)
    unique_layouts = []
    for layout in layouts:
        if layout and isinstance(layout, dict):
            # Simple deduplication based on space_type only
            layout_key = layout.get("space_type", f"layout_{len(unique_layouts)}")
            if not any(ul.get("key") == layout_key for ul in unique_layouts):
                unique_layouts.append({"key": layout_key, "output": layout})

    for i, layout_info in enumerate(unique_layouts):
        layout_cid = generate_placeholder_cid("layout", f"{property_cid}_layout_{layout_info['key']}_{i}")
        relationships.append(create_relationship(property_cid, layout_cid, "property_has_layout"))

        # REMOVED: Layout to image relationships as requested
        # Link layout to ALL images in the batch (since we can't determine which specific image shows this layout)
        # In a more sophisticated system, you might analyze image content to determine this
        # for image_info in image_cids.values():
        #     relationships.append(create_relationship(layout_cid, image_info["cid"], "layout_has_image"))

        # Handle appliances in this layout
        layout_data = layout_info["output"]
        layout_appliances = layout_data.get("appliances", [])
        if isinstance(layout_appliances, dict):
            layout_appliances = [layout_appliances]
        elif not isinstance(layout_appliances, list):
            layout_appliances = []

        for j, appliance in enumerate(layout_appliances):
            if appliance and isinstance(appliance, dict):
                appliance_cid = generate_placeholder_cid("appliance", f"{layout_cid}_appliance_{j}")
                relationships.append(create_relationship(layout_cid, appliance_cid, "layout_has_appliance"))

    # Property-level appliances (separate from layout appliances)
    appliances = batch_data.get("appliance", [])
    if isinstance(appliances, dict):
        appliances = [appliances]
    elif not isinstance(appliances, list):
        appliances = []

    for i, appliance in enumerate(appliances):
        if appliance and isinstance(appliance, dict):
            appliance_cid = generate_placeholder_cid("appliance", f"{property_cid}_appliance_{i}")
            relationships.append(create_relationship(property_cid, appliance_cid, "property_has_appliance"))

    # Return both relationships and metadata for debugging
    return {
        "relationships": relationships,
        "metadata": {
            "property_cid": property_cid,
            "image_count": len(image_cids),
            "layout_count": len(unique_layouts),
            "appliance_count": len(appliances),
            "property_objects": list(property_level_objects.keys())
        }
    }


def generate_relationships_per_image(extracted_data, image_name, property_cid, image_index):
    """Generate relationships for a single image's extracted output."""
    relationships = []

    # Create CID for this specific image
    image_cid = generate_placeholder_cid("image", f"{image_name.replace('.', '_')}_{image_index}")

    # Link image to property
    relationships.append(create_relationship(property_cid, image_cid, "property_has_file"))

    # Handle layouts found in this specific image
    layouts = extracted_data.get("layout", [])
    if isinstance(layouts, dict):
        layouts = [layouts]
    elif not isinstance(layouts, list):
        layouts = []

    for i, layout in enumerate(layouts):
        if layout and isinstance(layout, dict):
            # Create layout CID specific to this image
            layout_cid = generate_placeholder_cid("layout", f"{property_cid}_img{image_index}_layout_{i}")
            relationships.append(create_relationship(property_cid, layout_cid, "property_has_layout"))
            # REMOVED: Layout to image relationships as requested
            # relationships.append(create_relationship(layout_cid, image_cid, "layout_has_image"))

            # Handle appliances in this layout
            layout_appliances = layout.get("appliances", [])
            if isinstance(layout_appliances, dict):
                layout_appliances = [layout_appliances]
            elif not isinstance(layout_appliances, list):
                layout_appliances = []

            for j, appliance in enumerate(layout_appliances):
                if appliance and isinstance(appliance, dict):
                    appliance_cid = generate_placeholder_cid("appliance", f"{layout_cid}_appliance_{j}")
                    relationships.append(create_relationship(layout_cid, appliance_cid, "layout_has_appliance"))

    # Handle appliances found in this specific image (not in layouts)
    appliances = extracted_data.get("appliance", [])
    if isinstance(appliances, dict):
        appliances = [appliances]
    elif not isinstance(appliances, list):
        appliances = []

    for i, appliance in enumerate(appliances):
        if appliance and isinstance(appliance, dict):
            appliance_cid = generate_placeholder_cid("appliance", f"{property_cid}_img{image_index}_appliance_{i}")
            relationships.append(create_relationship(property_cid, appliance_cid, "property_has_appliance"))

    return relationships


def chunk_list(lst, chunk_size):
    """Split list into chunks of `chunk_size`."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def get_openai_cost_for_today(api_key=None):
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("[ERROR] No API key found in environment.")
        return 0.0

    today = datetime.utcnow().date().isoformat()
    url = "https://api.openai.com/v1/dashboard/billing/usage"
    params = {
        "start_date": today,
        "end_date": today
    }
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        usage = data.get("total_usage", 0) / 100.0
        logger.info(f"\n📊 OpenAI usage for API key on {today}: ${usage:.4f}")
        return usage
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch usage: {e}")
        return 0.0


def optimize_image(image_path):
    """Optimize image by converting to WebP and reducing resolution."""
    try:
        # Initialize image converter
        converter = ImageConverter(logger)

        with Image.open(image_path) as img:
            # Get original size for logging
            original_size = os.path.getsize(image_path) / 1024  # KB

            # Resize if needed
            if img.width > MAX_IMAGE_SIZE[0] or img.height > MAX_IMAGE_SIZE[1]:
                img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

            # Convert to WebP for better compression
            try:
                b64_data = converter.convert_to_webp_base64(img, quality=85)

                # Log compression results
                webp_size = len(base64.b64decode(b64_data)) / 1024  # KB
                compression_ratio = (1 - webp_size / original_size) * 100
                logger.info(f"Optimized {os.path.basename(image_path)}: "
                          f"{original_size:.1f}KB -> {webp_size:.1f}KB "
                          f"({compression_ratio:.1f}% reduction)")

                return b64_data

            except Exception as webp_error:
                logger.warning(f"WebP conversion failed, falling back to JPEG: {webp_error}")
                # Fallback to JPEG
                if img.mode != "RGB":
                    img = img.convert("RGB")

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
                buffer.seek(0)

                return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception as e:
        logger.error(f"[ERROR] Failed to optimize image {image_path}: {e}")
        return encode_image_original(image_path)


def encode_image_original(image_path):
    """Original image encoding method as fallback."""
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def optimize_s3_image(s3_key):
    """Download and optimize an S3 image, then clean up the temp file."""
    temp_path = None
    try:
        # Download image to temp file
        temp_path = download_s3_image_to_temp(s3_key)
        if not temp_path:
            logger.error(f"[ERROR] Failed to download S3 image: {s3_key}")
            return None

        # Optimize the image
        optimized_b64 = optimize_image(temp_path)
        if not optimized_b64:
            logger.error(f"[ERROR] Failed to optimize image: {s3_key}")
            return None

        return optimized_b64
    except Exception as e:
        logger.error(f"[ERROR] Failed to optimize S3 image {s3_key}: {e}")
        return None
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as cleanup_error:
                logger.error(f"[WARNING] Failed to cleanup temp file {temp_path}: {cleanup_error}")
                pass





def load_schemas_from_ipfs():
    """Load all schemas from IPFS using the hardcoded CIDs."""
    schemas = {}

    logger.info("Loading schemas from IPFS...")

    for schema_name, cid in IPFS_SCHEMA_CIDS.items():
        logger.info(f"Fetching {schema_name} schema from IPFS CID: {cid}")
        schema = fetch_schema_from_ipfs(cid)
        if schema:
            schemas[schema_name] = schema
            logger.info(f"✓ Successfully loaded {schema_name} schema")
        else:
            logger.error(f"✗ Failed to load {schema_name} schema from IPFS")
            # Fallback to empty schema
            schemas[schema_name] = {}

    # Also load relationship schema
    logger.info(f"Fetching relationship schema from IPFS CID: {RELATIONSHIP_SCHEMA_CID}")
    relationship_schema = fetch_schema_from_ipfs(RELATIONSHIP_SCHEMA_CID)
    if relationship_schema:
        schemas["relationship"] = relationship_schema
        logger.info("✓ Successfully loaded relationship schema")
    else:
        logger.error("✗ Failed to load relationship schema from IPFS")
        schemas["relationship"] = {}

    return schemas

def load_optimized_json_schema_prompt(folder_name=None, schemas=None, county_layouts=None):
    """Optimized prompt with reduced token count while maintaining functionality."""
    if schemas is None:
        # Fallback to local files if IPFS schemas not provided
        combined_schema = {}
        for key in SCHEMA_KEYS:
            path = os.path.join(SCHEMA_FOLDER, f"{key}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    combined_schema[key] = json.load(f)
            else:
                combined_schema[key] = {}
    else:
        # Use IPFS schemas
        combined_schema = {}
        for key in SCHEMA_KEYS:
            combined_schema[key] = schemas.get(key, {})

    # Enhanced categorization instructions
    categorization_instructions = """
IMPORTANT LAYOUT DETECTION AND GROUPING RULES:
- All images in this batch are from the same property/location: {folder_name}
- CAREFULLY ANALYZE EACH IMAGE to determine if it belongs to the same room/layout or a different one
- For multiple rooms of the same type (e.g., multiple bedrooms), create SEPARATE layout entries:
  * If images show different bedrooms, create separate layout entries
  * If images show different angles of the SAME bedroom, combine them into ONE layout entry
- Use these clues to determine if images are from the same room:
  * Wall color, flooring, furniture arrangement, window placement, room size
  * Different furniture, different wall colors, different layouts = different rooms
  * Same furniture, same wall colors, same layout = same room
- For each unique room/layout, create a separate layout entry
- Use the exact field names and structure from the IPFS layout schema
- Follow the schema properties exactly as defined
- Appliances found in the same room should be grouped under the same layout
- Structure and lot information should be consistent across all images from the same property
- Use consistent space_type names but differentiate with descriptive room names
""".format(folder_name=folder_name or "this property")

    # Build schema-specific instructions
    schema_instructions = ""
    for schema_key, schema_data in combined_schema.items():
        if schema_data and isinstance(schema_data, dict):
            properties = schema_data.get('properties', {})
            if properties:
                schema_instructions += f"\n{schema_key.upper()} SCHEMA:\n"
                for prop_name, prop_info in properties.items():
                    if isinstance(prop_info, dict):
                        description = prop_info.get('description', '')
                        prop_type = prop_info.get('type', '')
                        schema_instructions += f"- {prop_name}: {prop_type} - {description}\n"

                # Add specific instructions for layout schema
                if schema_key == "layout":
                    schema_instructions += """
LAYOUT SCHEMA FIELDS:
- Follow the exact field structure from the IPFS schema
- Use the fields defined in the schema properties
"""

    # Use IPFS schemas to build the prompt
    if schemas and isinstance(schemas, dict):
        # Build schema-specific instructions from IPFS schemas
        schema_instructions = ""
        example_structure = {}

        for schema_key, schema_data in schemas.items():
            if schema_data and isinstance(schema_data, dict) and schema_key != "relationship":
                properties = schema_data.get('properties', {})
                if properties:
                    schema_instructions += f"\n{schema_key.upper()} SCHEMA:\n"
                    example_structure[schema_key] = {}

                    for prop_name, prop_info in properties.items():
                        if isinstance(prop_info, dict):
                            description = prop_info.get('description', '')
                            prop_type = prop_info.get('type', '')

                            # Check for enum values
                            enum_values = prop_info.get('enum', [])
                            if enum_values:
                                schema_instructions += f"- {prop_name}: {prop_type} - {description} (ENUM: {', '.join(map(str, enum_values))})\n"
                            else:
                                schema_instructions += f"- {prop_name}: {prop_type} - {description}\n"

                            # Add example value based on type and enum
                            if enum_values:
                                example_structure[schema_key][prop_name] = enum_values[0] if enum_values else "example_value"
                            elif prop_type == "string":
                                example_structure[schema_key][prop_name] = "example_value"
                            elif prop_type == "array":
                                example_structure[schema_key][prop_name] = ["example_item"]
                            elif prop_type == "number":
                                example_structure[schema_key][prop_name] = 0
                            elif prop_type == "boolean":
                                example_structure[schema_key][prop_name] = True

        # Add county data information to prompt if available
        county_data_section = ""
        if county_layouts and len(county_layouts) > 0:
            county_data_section = f"""

COUNTY DATA INFORMATION:
This property has the following rooms/spaces according to county records:
"""
            bedroom_count = 0
            bathroom_count = 0

            for i, layout in enumerate(county_layouts, 1):
                space_type = layout.get("space_type", "unknown")
                county_data_section += f"- Layout {i}: {space_type}\n"

                if space_type.lower() == "bedroom":
                    bedroom_count += 1
                elif space_type.lower() == "bathroom":
                    bathroom_count += 1

            # Add the actual county layout JSON data
            county_data_section += f"""

COUNTY LAYOUT JSON DATA:
Use this county data as reference for creating separate layout entries:

{json.dumps(county_layouts, indent=2)}
"""

            county_data_section += f"""
TOTAL: {bedroom_count} bedroom(s), {bathroom_count} bathroom(s)

IMPORTANT COUNTY DATA RULES:
- When analyzing images, MATCH them to the county data rooms/spaces
- If you see a bedroom, match it to one of the {bedroom_count} county bedroom records
- If you see a bathroom, match it to one of the {bathroom_count} county bathroom records
- CRITICAL: Create SEPARATE layout entries for each county bathroom/bedroom record
- If county data shows {bathroom_count} bathrooms, create {bathroom_count} separate bathroom layout entries
- If county data shows {bedroom_count} bedrooms, create {bedroom_count} separate bedroom layout entries
- Use the county data as a reference for room types and details
- If you cannot determine specific details from images, use the county data details
- Maintain consistency with county data while adding visual details from images
- For rooms not in county data (kitchen, living room, etc.), analyze normally
- DO NOT merge multiple county bathrooms into one layout entry
"""

        prompt = f"""Analyze these real estate images and return detailed JSON following the IPFS schemas:

{schema_instructions}{county_data_section}

CRITICAL SCHEMA RULES:
- LAYOUT schema: Only include room/layout related fields (space_type, room_name, room_description, etc.)
- STRUCTURE schema: Only include building structure fields (roof_type, construction_material, etc.)
- APPLIANCE schema: Only include appliance fields (appliance_type, finish, condition, etc.)
- UTILITY schema: Only include utility fields (utility_type, condition, etc.)
- LOT schema: Only include lot/landscape fields (size, driveway_material, landscaping, etc.)
- DO NOT mix fields between schemas (e.g., don't put pool fields in layout schema)

ENUM VALUE RULES:
- For fields with ENUM values listed, you MUST use ONLY those exact values
- Do not invent new values - use only the enum values provided
- If you cannot determine the exact enum value, use null (NOT a new value)
- If the value you see doesn't match any enum value exactly, use null
- Example: If pool_type has ENUM: ["in-ground", "above-ground", "none"], use only these values or null

ANALYSIS INSTRUCTIONS:
- Look for specific features in each image: furniture, appliances, room types, building materials, utilities
- For layouts: Identify room types (bedroom, kitchen, bathroom, living room, etc.) and describe what you see
- For structure: Look for roof types, construction materials, architectural style, exterior features
- For appliances: Identify any visible appliances (refrigerator, stove, washer, dryer, etc.) and their condition
- For utilities: Look for HVAC systems, electrical panels, plumbing fixtures, etc.
- For lot: Identify landscaping, driveway materials, lot size indicators, outdoor features
- CRITICAL: Use ONLY the enum values provided in the schema - do not invent new values
- CRITICAL: Match what you see to the exact enum values available

DETAILED ANALYSIS RULES:
- Examine each image carefully for visible features
- Describe specific details you can see (colors, materials, styles, conditions)
- If you see multiple rooms of the same type, create separate layout entries
- Be specific about what you observe rather than making assumptions
- Include actual visible features, not generic descriptions
- IMPORTANT: Even if you don't see certain features (like lot, structure, utilities, appliances), still include those keys with empty arrays or objects to maintain schema structure
- IMPORTANT: Use ONLY the fields defined in each IPFS schema - do not add fields that aren't in the schema

Example structure based on IPFS schemas:
{json.dumps(example_structure, indent=2)}

Return ONLY the JSON object with detailed analysis of visible features. Include ALL schema keys even if empty."""
    else:
        # Fallback to basic structure if IPFS schemas not available
        prompt = f"""Analyze these real estate images and return detailed JSON with the following structure:

{{
  "layout": [
    {{
      "space_type": "kitchen",
      "room_name": "Kitchen",
      "room_description": "Modern kitchen with granite countertops"
    }}
  ],
  "structure": {{
    "roof_type": "shingle",
    "construction_material": "wood",
    "style": "modern"
  }},
  "utility": [
    {{
      "utility_type": "HVAC",
      "condition": "good"
    }}
  ],
  "appliance": [
    {{
      "appliance_type": "refrigerator",
      "finish": "stainless steel",
      "condition": "excellent"
    }}
  ],
  "lot": {{
    "size": "5000 sq ft",
    "driveway_material": "concrete",
    "landscaping": "mature trees"
  }}
}}

DETAILED ANALYSIS INSTRUCTIONS:
- Look for specific features in each image: furniture, appliances, room types, building materials, utilities
- For layouts: Identify room types (bedroom, kitchen, bathroom, living room, etc.) and describe what you see
- For structure: Look for roof types, construction materials, architectural style, exterior features
- For appliances: Identify any visible appliances (refrigerator, stove, washer, dryer, etc.) and their condition
- For utilities: Look for HVAC systems, electrical panels, plumbing fixtures, etc.
- For lot: Identify landscaping, driveway materials, lot size indicators, outdoor features

ANALYSIS RULES:
- Examine each image carefully for visible features
- Describe specific details you can see (colors, materials, styles, conditions)
- If you see multiple rooms of the same type, create separate layout entries
- Be specific about what you observe rather than making assumptions
- Include actual visible features, not generic descriptions
- IMPORTANT: Even if you don't see certain features (like lot, structure, utilities, appliances), still include those keys with empty arrays or objects to maintain schema structure
- Only include what's visible in the images
- Use empty arrays [] if no items found

Return ONLY the JSON object with detailed analysis of visible features. Include ALL schema keys even if empty."""

    return prompt


def try_parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip().replace("```json", "").replace("```", "")
        try:
            return json.loads(cleaned)
        except Exception as e:
            logger.error(f"[ERROR] Could not parse JSON: {e}")
            logger.debug(f"[DEBUG] Text: {text[:300]}...")
            return None


def call_openai_optimized_s3(image_objects, prompt, schemas=None):
    """Optimized OpenAI call for local images with retry logic and better error handling."""
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
            dynamic_timeout = min(180, 30 + (batch_size * 15))  # 30s base + 15s per image, max 180s

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0,
                    timeout=dynamic_timeout
                )
            except Exception as api_error:
                logger.error(f"API error type: {type(api_error).__name__}")
                if "timeout" in str(api_error).lower():
                    logger.error(f"Request timed out after {dynamic_timeout} seconds for batch of {batch_size} images")
                raise api_error

            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

            prompt_cost = prompt_tokens * 0.005 / 1000
            completion_cost = completion_tokens * 0.015 / 1000
            total_cost = prompt_cost + completion_cost

            # Thread-safe update of global stats
            with stats_lock:
                TOTAL_PROMPT_TOKENS += prompt_tokens
                TOTAL_COMPLETION_TOKENS += completion_tokens
                TOTAL_COST += total_cost

            result = try_parse_json(response.choices[0].message.content)
            if result is None:
                logger.error(f"    [DEBUG] Failed to parse JSON from OpenAI response")
                logger.debug(f"    [DEBUG] Response content: {response.choices[0].message.content[:200]}...")
            else:
                # Debug: Show what the AI actually returned
                logger.debug(f"    [DEBUG] AI Response: {json.dumps(result, indent=2)[:500]}...")

                # Validate that result has the expected schema keys from IPFS
                if schemas and isinstance(schemas, dict):
                    expected_keys = [key for key in schemas.keys() if key != "relationship"]
                    missing_keys = [key for key in expected_keys if key not in result]
                    if missing_keys:
                        logger.debug(f"    [DEBUG] Missing IPFS schema keys: {missing_keys}")
                        # Add missing keys with appropriate default values
                        for key in missing_keys:
                            schema_data = schemas.get(key, {})
                            if schema_data and isinstance(schema_data, dict):
                                # Check if it should be an array or object based on schema
                                if schema_data.get('type') == 'array':
                                    result[key] = []
                                else:
                                    result[key] = {}
                            else:
                                # Default to array for most schemas
                                result[key] = []
                    logger.debug(f"    [DEBUG] Generated JSON structure: {list(result.keys())}")

                    # Validate schema field compliance
                    for schema_key, schema_data in schemas.items():
                        if schema_key != "relationship" and schema_key in result:
                            schema_properties = schema_data.get("properties", {})
                            if schema_properties:
                                # Check if result contains fields not in schema
                                invalid_fields = []
                                if isinstance(result[schema_key], dict):
                                    for field in result[schema_key].keys():
                                        if field not in schema_properties:
                                            invalid_fields.append(field)
                                elif isinstance(result[schema_key], list) and result[schema_key]:
                                    # Check first item in array
                                    if isinstance(result[schema_key][0], dict):
                                        for field in result[schema_key][0].keys():
                                            if field not in schema_properties:
                                                invalid_fields.append(field)

                                if invalid_fields:
                                    logger.debug(f"    [DEBUG] Invalid fields in {schema_key}: {invalid_fields}")
                                    # Remove invalid fields
                                    if isinstance(result[schema_key], dict):
                                        for field in invalid_fields:
                                            result[schema_key].pop(field, None)
                                    elif isinstance(result[schema_key], list) and result[schema_key]:
                                        for item in result[schema_key]:
                                            if isinstance(item, dict):
                                                for field in invalid_fields:
                                                    item.pop(field, None)

                                        # Also remove any fields that don't belong in layout schema
                                        layout_schema_properties = schema_data.get("properties", {})
                                        if layout_schema_properties:
                                            for field in list(result[schema_key].keys()) if isinstance(result[schema_key], dict) else []:
                                                if field not in layout_schema_properties:
                                                    result[schema_key].pop(field, None)
                                                    logger.debug(f"    [DEBUG] Removed invalid field '{field}' from layout")
                                            if isinstance(result[schema_key], list) and result[schema_key]:
                                                for item in result[schema_key]:
                                                    if isinstance(item, dict):
                                                        for field in list(item.keys()):
                                                            if field not in layout_schema_properties:
                                                                item.pop(field, None)
                                                                logger.debug(f"    [DEBUG] Removed invalid field '{field}' from layout item")
                else:
                    # Fallback validation
                    expected_keys = ['layout', 'structure', 'utility', 'appliance', 'lot']
                    missing_keys = [key for key in expected_keys if key not in result]
                    if missing_keys:
                        logger.debug(f"    [DEBUG] Missing schema keys: {missing_keys}")
                        for key in missing_keys:
                            if key in ['layout', 'utility', 'appliance']:
                                result[key] = []
                            else:
                                result[key] = {}
                    logger.debug(f"    [DEBUG] Generated JSON structure: {list(result.keys())}")
            return result, total_cost

        except Exception as e:
            logger.error(f"    [DEBUG] API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                logger.debug(f"    [DEBUG] Retrying in {retry_delay} seconds...")
            else:
                logger.error(f"    [DEBUG] All {max_retries} attempts failed")
                return None, 0.0

    return None, 0.0


def call_openai_optimized(image_paths, prompt):
    """Optimized OpenAI call with reduced system message and better batching."""
    global TOTAL_IMAGES_PROCESSED, TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, TOTAL_COST

    try:
        images_in_batch = len(image_paths[:10])

        # Thread-safe update of global stats
        with stats_lock:
            TOTAL_IMAGES_PROCESSED += images_in_batch

        messages = [
            {"role": "system", "content": "Real estate image analyzer. Return JSON only."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]

        for image_path in image_paths[:10]:
            image_b64 = optimize_image(image_path)
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })

        # Calculate dynamic timeout based on batch size
        batch_size = len(image_paths[:10])
        dynamic_timeout = min(120, 30 + (batch_size * 15))  # 30s base + 15s per image, max 120s

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            timeout=dynamic_timeout
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        prompt_cost = prompt_tokens * 0.005 / 1000
        completion_cost = completion_tokens * 0.015 / 1000
        total_cost = prompt_cost + completion_cost

        # Thread-safe update of global stats
        with stats_lock:
            TOTAL_PROMPT_TOKENS += prompt_tokens
            TOTAL_COMPLETION_TOKENS += completion_tokens
            TOTAL_COST += total_cost

        logger.info(f"[TOKENS] Prompt: {prompt_tokens}, Completion: {completion_tokens}")
        logger.info(f"[COST] ${total_cost:.6f} | Images in batch: {images_in_batch}")

        return try_parse_json(response.choices[0].message.content), total_cost

    except Exception as e:
        image_names = ", ".join(os.path.basename(p) for p in image_paths)
        logger.error(f"[ERROR] API failed for batch: {image_names}\nReason: {e}")
        return None, 0.0


def generate_image_json_files_s3(image_objects, output_dir, batch_number):
    """Generate individual JSON files for each S3 image using the file schema from IPFS."""
    image_files = {}
    converter = ImageConverter(logger)

    # Images will be saved in the same folder as JSON files

    # Fetch file schema from IPFS
    file_schema = fetch_schema_from_ipfs(IPFS_SCHEMA_CIDS["file"])
    if not file_schema:
        logger.error(f"Could not fetch file schema from IPFS CID {IPFS_SCHEMA_CIDS['file']}")
        logger.error("Exiting due to missing file schema")
        sys.exit(1)

    for i, image_obj in enumerate(image_objects):
        # Generate a unique CID for the image
        original_name = image_obj['name']
        # Change extension to .webp
        base_name = os.path.splitext(original_name)[0]
        webp_name = f"{base_name}.webp"
        image_cid = generate_clean_cid("file", webp_name.replace(".", "_"))

        # Download image from S3 to temp location first
        temp_path = os.path.join(output_dir, f"temp_{original_name}")
        webp_dest_path = os.path.join(output_dir, webp_name)
        
        try:
            s3_key = image_obj['key']
            s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_path)
            logger.info(f"Downloaded image: {original_name} from S3")
            
            # Convert to WebP
            webp_bytes = converter.convert_to_webp(temp_path, quality=90)
            with open(webp_dest_path, 'wb') as f:
                f.write(webp_bytes)
            logger.info(f"Converted to WebP: {original_name} -> {webp_name}")
            
            # Remove temp file
            os.unlink(temp_path)
            
        except Exception as e:
            logger.error(f"Failed to download/convert image {original_name}: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            continue

        # Create image JSON using the file schema from IPFS with relative path
        relative_image_path = f"./{webp_name}"

        # Use IPFS schema structure directly (no type/properties wrapper)
        image_data = {}

        # Populate properties based on IPFS schema
        if file_schema and isinstance(file_schema, dict):
            schema_properties = file_schema.get("properties", {})
            if schema_properties:
                for prop_name, prop_info in schema_properties.items():
                    if isinstance(prop_info, dict):
                        prop_type = prop_info.get("type", "string")

                        # Set appropriate values based on property name
                        if prop_name == "ipfs_url":
                            image_data[prop_name] = relative_image_path
                        elif prop_name == "name":
                            image_data[prop_name] = webp_name
                        elif prop_name == "file_format":
                            image_data[prop_name] = "webp"
                        elif prop_name == "document_type":
                            image_data[prop_name] = "image"
                        else:
                            # Set default values based on type
                            if prop_type == "string":
                                image_data[prop_name] = ""
                            elif prop_type == "number":
                                image_data[prop_name] = 0
                            elif prop_type == "boolean":
                                image_data[prop_name] = False
                            else:
                                image_data[prop_name] = None

        # Save individual image file
        filename = f"{image_cid}_photo_metadata.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(image_data, f, indent=2)

        image_files[image_cid] = filename
        logger.info(f"Saved: {filename}")

    return image_files


def generate_image_json_files(image_paths, output_dir, batch_number):
    """Generate individual JSON files for each image using the file schema from IPFS."""
    image_files = {}
    converter = ImageConverter(logger)

    # Images will be saved in the same folder as JSON files

    # Fetch file schema from IPFS
    file_schema = fetch_schema_from_ipfs(IPFS_SCHEMA_CIDS["file"])
    if not file_schema:
        logger.error(f"Could not fetch file schema from IPFS CID {IPFS_SCHEMA_CIDS['file']}")
        logger.error("Exiting due to missing file schema")
        sys.exit(1)

    for i, image_path in enumerate(image_paths):
        # Generate a unique CID for the image
        original_name = os.path.basename(image_path)
        # Change extension to .webp
        base_name = os.path.splitext(original_name)[0]
        webp_name = f"{base_name}.webp"
        image_cid = generate_clean_cid("file", webp_name.replace(".", "_"))

        # Convert and save as WebP to output folder
        webp_dest_path = os.path.join(output_dir, webp_name)
        try:
            # Convert to WebP
            webp_bytes = converter.convert_to_webp(image_path, quality=90)
            with open(webp_dest_path, 'wb') as f:
                f.write(webp_bytes)
            logger.info(f"Converted and saved: {original_name} -> {webp_name}")
        except Exception as e:
            logger.error(f"Failed to convert image {original_name}: {e}")
            continue

        # Create image JSON using the file schema from IPFS with relative path
        relative_image_path = f"./{webp_name}"

        # Use IPFS schema structure directly (no type/properties wrapper)
        image_data = {}

        # Populate properties based on IPFS schema
        if file_schema and isinstance(file_schema, dict):
            schema_properties = file_schema.get("properties", {})
            if schema_properties:
                for prop_name, prop_info in schema_properties.items():
                    if isinstance(prop_info, dict):
                        prop_type = prop_info.get("type", "string")

                        # Set appropriate values based on property name
                        if prop_name == "ipfs_url":
                            image_data[prop_name] = relative_image_path
                        elif prop_name == "name":
                            image_data[prop_name] = webp_name
                        elif prop_name == "file_format":
                            image_data[prop_name] = "webp"
                        elif prop_name == "document_type":
                            image_data[prop_name] = "image"
                        else:
                            # Set default values based on type
                            if prop_type == "string":
                                image_data[prop_name] = ""
                            elif prop_type == "number":
                                image_data[prop_name] = 0
                            elif prop_type == "boolean":
                                image_data[prop_name] = False
                            else:
                                image_data[prop_name] = None

        # Save individual image file
        filename = f"{image_cid}_photo_metadata.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(image_data, f, indent=2)

        image_files[image_cid] = filename
        logger.info(f"Saved: {filename}")

    return image_files


def generate_individual_object_files_s3(batch_data, image_objects, output_dir, batch_number):
    """
    Generate individual JSON files for each space_type layout and appliance type for S3 images.
    Returns a mapping of object types to their file names for relationship generation.
    """
    object_files = {
        "layouts": {},  # space_type -> filename
        "appliances": {},  # appliance_type -> filename
        "property_objects": {},  # structure, lot, etc. -> filename
        "images": {}  # image_cid -> filename
    }

    # Generate image files first
    logger.info(f"Generating image files...")
    image_files = generate_image_json_files_s3(image_objects, output_dir, batch_number)
    object_files["images"] = image_files

    # Generate property-level object files
    property_objects = ["structure", "lot", "utility", "nearby_location"]
    for obj_type in property_objects:
        if batch_data.get(obj_type):
            filename = f"{obj_type}_batch_{batch_number:02d}_photo_metadata.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w") as f:
                json.dump(batch_data[obj_type], f, indent=2)

            object_files["property_objects"][obj_type] = filename
            logger.info(f"    [✔] Saved: {filename}")

    # Generate layout files per space_type
    layouts = batch_data.get("layout", [])
    if isinstance(layouts, dict):
        layouts = [layouts]
    elif not isinstance(layouts, list):
        layouts = []

    layout_counter = {}  # Track multiple instances of same space_type
    for layout in layouts:
        if layout and isinstance(layout, dict):
            space_type = layout.get("space_type", "unknown_space")

            # Handle case where space_type might be a dict or other non-string type
            if isinstance(space_type, dict):
                # If it's a dict, try to get a meaningful identifier
                space_type = space_type.get("name", space_type.get("type", "unknown_space"))
            elif not isinstance(space_type, str):
                # Convert to string if it's not a string
                space_type = str(space_type)

            space_type = space_type.lower().replace(" ", "_")

            # Handle multiple instances of same space_type
            if space_type in layout_counter:
                layout_counter[space_type] += 1
                instance_suffix = f"_{layout_counter[space_type]}"
            else:
                layout_counter[space_type] = 1
                instance_suffix = ""

            filename = f"layout_{space_type}{instance_suffix}_batch_{batch_number:02d}.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w") as f:
                json.dump(layout, f, indent=2)

            object_files["layouts"][f"{space_type}{instance_suffix}"] = filename
            logger.info(f"    [✔] Saved: {filename}")

    # Generate appliance files per appliance type
    appliances = batch_data.get("appliance", [])
    if isinstance(appliances, dict):
        appliances = [appliances]
    elif not isinstance(appliances, list):
        appliances = []

    appliance_counter = {}  # Track multiple instances of same appliance_type
    for appliance in appliances:
        if appliance and isinstance(appliance, dict):
            appliance_type = appliance.get("appliance_type", appliance.get("type", "unknown"))

            # Handle case where appliance_type might be a dict or other non-string type
            if isinstance(appliance_type, dict):
                # If it's a dict, try to get a meaningful identifier
                appliance_type = appliance_type.get("name", appliance_type.get("type", "unknown"))
            elif not isinstance(appliance_type, str):
                # Convert to string if it's not a string
                appliance_type = str(appliance_type)

            appliance_type = appliance_type.lower().replace(" ", "_")

            # Handle multiple instances of same appliance_type
            if appliance_type in appliance_counter:
                appliance_counter[appliance_type] += 1
                instance_suffix = f"_{appliance_counter[appliance_type]}"
            else:
                appliance_counter[appliance_type] = 1
                instance_suffix = ""

            filename = f"appliance_{appliance_type}{instance_suffix}_batch_{batch_number:02d}.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w") as f:
                json.dump(appliance, f, indent=2)

            object_files["appliances"][f"{appliance_type}{instance_suffix}"] = filename
            logger.info(f"    [✔] Saved: {filename}")

    return object_files


def generate_individual_object_files(batch_data, image_paths, output_dir, batch_number):
    """
    Generate individual JSON files for each space_type layout and appliance type.
    Returns a mapping of object types to their file names for relationship generation.
    """
    object_files = {
        "layouts": {},  # space_type -> filename
        "appliances": {},  # appliance_type -> filename
        "property_objects": {},  # structure, lot, etc. -> filename
        "images": {}  # image_cid -> filename
    }

    # Generate image files first
    logger.info(f"Generating image files...")
    image_files = generate_image_json_files(image_paths, output_dir, batch_number)
    object_files["images"] = image_files
    logger.info(f"Generated {len(image_files)} image files")

    # Generate property-level object files
    property_objects = ["structure", "lot", "utility", "nearby_location"]
    for obj_type in property_objects:
        if batch_data.get(obj_type):
            filename = f"{obj_type}_batch_{batch_number:02d}_photo_metadata.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w") as f:
                json.dump(batch_data[obj_type], f, indent=2)

            object_files["property_objects"][obj_type] = filename
            logger.info(f"    [✔] Saved: {filename}")

    # Generate layout files per space_type
    layouts = batch_data.get("layout", [])
    if isinstance(layouts, dict):
        layouts = [layouts]
    elif not isinstance(layouts, list):
        layouts = []

    layout_counter = {}  # Track multiple instances of same space_type
    for layout in layouts:
        if layout and isinstance(layout, dict):
            space_type = layout.get("space_type", "unknown_space")

            # Handle case where space_type might be a dict or other non-string type
            if isinstance(space_type, dict):
                # If it's a dict, try to get a meaningful identifier
                space_type = space_type.get("name", space_type.get("type", "unknown_space"))
            elif not isinstance(space_type, str):
                # Convert to string if it's not a string
                space_type = str(space_type)

            space_type = space_type.lower().replace(" ", "_")

            # Handle multiple instances of same space_type
            if space_type in layout_counter:
                layout_counter[space_type] += 1
                instance_suffix = f"_{layout_counter[space_type]}"
            else:
                layout_counter[space_type] = 1
                instance_suffix = ""

            filename = f"layout_{space_type}{instance_suffix}_batch_{batch_number:02d}.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w") as f:
                json.dump(layout, f, indent=2)

            object_files["layouts"][f"{space_type}{instance_suffix}"] = filename
            logger.info(f"    [✔] Saved: {filename}")

    # Generate appliance files per appliance type
    appliances = batch_data.get("appliance", [])
    if isinstance(appliances, dict):
        appliances = [appliances]
    elif not isinstance(appliances, list):
        appliances = []

    appliance_counter = {}  # Track multiple instances of same appliance_type
    for appliance in appliances:
        if appliance and isinstance(appliance, dict):
            appliance_type = appliance.get("appliance_type", appliance.get("type", "unknown"))

            # Handle case where appliance_type might be a dict or other non-string type
            if isinstance(appliance_type, dict):
                # If it's a dict, try to get a meaningful identifier
                appliance_type = appliance_type.get("name", appliance_type.get("type", "unknown"))
            elif not isinstance(appliance_type, str):
                # Convert to string if it's not a string
                appliance_type = str(appliance_type)

            appliance_type = appliance_type.lower().replace(" ", "_")

            # Handle multiple instances of same appliance_type
            if appliance_type in appliance_counter:
                appliance_counter[appliance_type] += 1
                instance_suffix = f"_{appliance_counter[appliance_type]}"
            else:
                appliance_counter[appliance_type] = 1
                instance_suffix = ""

            filename = f"appliance_{appliance_type}{instance_suffix}_batch_{batch_number:02d}.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w") as f:
                json.dump(appliance, f, indent=2)

            object_files["appliances"][f"{appliance_type}{instance_suffix}"] = filename
            logger.info(f"    [✔] Saved: {filename}")

    return object_files


def generate_clean_cid(object_type, identifier):
    """Generate a clean CID using just the meaningful name without UUID metadata."""
    # Clean the identifier to be filesystem/CID safe
    clean_id = identifier.replace("/", "_").replace("\\", "_").replace(":", "").replace(" ", "_").replace(".", "_")
    return f"{object_type}_{clean_id}"


def generate_relationships_from_object_files_s3(object_files, image_objects, property_cid, batch_number, property_id, relationship_schema=None):
    """
    Generate relationships using clean, meaningful CIDs based on filenames for S3 images.
    Now uses actual filenames as CIDs in relationships and IPFS relationship schema.
    Handles both new files and updated files to avoid duplicate relationships.
    """
    relationships = []

    # Get image CIDs from the generated image files (these are already clean filenames)
    image_cids = object_files["images"]  # This is a dict: {image_cid -> filename}

    # Create property -> image relationships using image CIDs (which are filenames without .json)
    for image_cid in image_cids.keys():
        relationships.append(create_relationship(property_cid, image_cid, "property_has_file", relationship_schema))

    # Property-level object relationships using filenames as CIDs
    # Only create relationships for new files, not updated ones
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
    for layout_key, filename in object_files["layouts"].items():
        # Use filename without .json extension as CID
        layout_cid = filename.replace(".json", "")
        relationships.append(create_relationship(property_cid, layout_cid, "property_has_layout", relationship_schema))

        # REMOVED: Layout to image relationships as requested
        # Link layout to all images in batch
        # for image_cid in image_cids.keys():
        #     relationships.append(create_relationship(layout_cid, image_cid, "layout_has_image", relationship_schema))

    # Appliance relationships using filenames as CIDs
    for appliance_key, filename in object_files["appliances"].items():
        # Use filename without .json extension as CID
        appliance_cid = filename.replace(".json", "")
        relationships.append(create_relationship(property_cid, appliance_cid, "property_has_appliance", relationship_schema))

    return relationships


def generate_relationships_from_object_files(object_files, image_paths, property_cid, batch_number, folder_path, relationship_schema=None):
    """
    Generate relationships using clean, meaningful CIDs based on filenames.
    Now uses actual filenames as CIDs in relationships.
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
    for layout_key, filename in object_files["layouts"].items():
        # Use filename without .json extension as CID
        layout_cid = filename.replace(".json", "")
        relationships.append(create_relationship(property_cid, layout_cid, "property_has_layout", relationship_schema))

        # REMOVED: Layout to image relationships as requested
        # Link layout to all images in batch
        # for image_cid in image_cids.keys():
        #     relationships.append(create_relationship(layout_cid, image_cid, "layout_has_image"))

    # Appliance relationships using filenames as CIDs
    for appliance_key, filename in object_files["appliances"].items():
        # Use filename without .json extension as CID
        appliance_cid = filename.replace(".json", "")
        relationships.append(create_relationship(property_cid, appliance_cid, "property_has_appliance", relationship_schema))

    return relationships


def merge_and_update_object_files_s3(batch_data, image_objects, output_dir, batch_number):
    """
    Generate or update JSON files for S3 images, merging data instead of creating new files.
    Returns a mapping of object types to their file names for relationship generation.
    """
    object_files = {
        "layouts": {},  # space_type -> filename
        "appliances": {},  # appliance_type -> filename
        "property_objects": {},  # structure, lot, etc. -> filename
        "images": {}  # image_cid -> filename
    }

    # Generate image files first (these are always new per batch)
    logger.info(f"    [→] Generating image files...")
    image_files = generate_image_json_files_s3(image_objects, output_dir, batch_number)
    object_files["images"] = image_files

    # Handle property-level objects (structure, lot, utility) - merge/update existing
    property_objects = ["structure", "lot", "utility"]
    for obj_type in property_objects:
        if batch_data.get(obj_type):
            filename = f"{obj_type}_photo_metadata.json"
            filepath = os.path.join(output_dir, filename)

            # Load existing data if file exists
            existing_data = None
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r") as f:
                        existing_data = json.load(f)
                    logger.info(f"    [→] Found existing {obj_type} data, merging...")
                except Exception as e:
                    logger.error(f"    [!] Error reading existing {obj_type} file: {e}")
                    existing_data = None

            # Merge new data with existing data
            new_data = batch_data[obj_type]
            if existing_data:
                # Merge logic depends on the object type
                if obj_type == "structure":
                    # For structure, merge properties
                    merged_data = merge_structure_data(existing_data, new_data)
                elif obj_type == "lot":
                    # For lot, merge properties
                    merged_data = merge_lot_data(existing_data, new_data)
                elif obj_type == "utility":
                    # For utility, merge arrays
                    merged_data = merge_utility_data(existing_data, new_data)
                else:
                    # Default merge - replace with new data
                    merged_data = new_data
            else:
                merged_data = new_data

            # Save merged data
            with open(filepath, "w") as f:
                json.dump(merged_data, f, indent=2)

            object_files["property_objects"][obj_type] = filename
            logger.info(f"    [✔] {'Updated' if existing_data else 'Saved'}: {filename}")

    # Handle layouts - create separate files for different rooms of the same type
    layouts = batch_data.get("layout", [])
    if isinstance(layouts, dict):
        layouts = [layouts]
    elif not isinstance(layouts, list):
        layouts = []

    # Handle layouts - create separate files for different rooms of the same type
    for layout in layouts:
        if layout and isinstance(layout, dict):
            space_type = layout.get("space_type", "unknown_space")

            # Handle case where space_type might be a dict or other non-string type
            if isinstance(space_type, dict):
                space_type = space_type.get("name", space_type.get("type", "unknown_space"))
            elif not isinstance(space_type, str):
                space_type = str(space_type)

            space_type = space_type.lower().replace(" ", "_")

            # Extract descriptive identifier from layout data to differentiate rooms
            layout_identifier = layout.get("layout_identifier", "")
            room_description = layout.get("room_description", "")
            room_name = layout.get("room_name", "")

            # Create a unique identifier for this specific layout
            if layout_identifier:
                unique_id = layout_identifier.lower().replace(" ", "_")
            elif room_description:
                unique_id = room_description.lower().replace(" ", "_")[:20]
            elif room_name:
                unique_id = room_name.lower().replace(" ", "_")
            else:
                # If no specific identifier, use a timestamp-based one
                unique_id = f"layout_{int(time.time())}"

            # Create filename with descriptive identifier
            if unique_id and unique_id != space_type:
                filename = f"layout_{space_type}_{unique_id}_photo_metadata.json"
            else:
                filename = f"layout_{space_type}_photo_metadata.json"

            filepath = os.path.join(output_dir, filename)

            # Load existing layout data if file exists
            existing_layout = None
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r") as f:
                        existing_layout = json.load(f)
                    logger.info(f"    [→] Found existing layout for {space_type} ({unique_id}), merging...")
                except Exception as e:
                    logger.error(f"    [!] Error reading existing layout file: {e}")
                    existing_layout = None

            # Merge layout data
            if existing_layout:
                merged_layout = merge_layout_data(existing_layout, layout)
            else:
                merged_layout = layout

            # Save merged layout
            with open(filepath, "w") as f:
                json.dump(merged_layout, f, indent=2)

            # Store in object_files with the unique identifier as key
            layout_key = f"{space_type}_{unique_id}" if unique_id != space_type else space_type
            object_files["layouts"][layout_key] = filename
            logger.info(f"    [✔] {'Updated' if existing_layout else 'Saved'}: {filename}")



    # Handle appliances - merge by appliance_type
    appliances = batch_data.get("appliance", [])
    if isinstance(appliances, dict):
        appliances = [appliances]
    elif not isinstance(appliances, list):
        appliances = []

    for appliance in appliances:
        if appliance and isinstance(appliance, dict):
            appliance_type = appliance.get("appliance_type", appliance.get("type", "unknown"))

            # Handle case where appliance_type might be a dict or other non-string type
            if isinstance(appliance_type, dict):
                appliance_type = appliance_type.get("name", appliance_type.get("type", "unknown"))
            elif not isinstance(appliance_type, str):
                appliance_type = str(appliance_type)

            appliance_type = appliance_type.lower().replace(" ", "_")
            filename = f"appliance_{appliance_type}_photo_metadata.json"
            filepath = os.path.join(output_dir, filename)

            # Load existing appliance data if file exists
            existing_appliance = None
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r") as f:
                        existing_appliance = json.load(f)
                    logger.info(f"    [→] Found existing appliance {appliance_type}, merging...")
                except Exception as e:
                    logger.error(f"    [!] Error reading existing appliance file: {e}")
                    existing_appliance = None

            # Merge appliance data
            if existing_appliance:
                merged_appliance = merge_appliance_data(existing_appliance, appliance)
            else:
                merged_appliance = appliance

            # Save merged appliance
            with open(filepath, "w") as f:
                json.dump(merged_appliance, f, indent=2)

            object_files["appliances"][appliance_type] = filename
            logger.info(f"    [✔] {'Updated' if existing_appliance else 'Saved'}: {filename}")

    return object_files


def merge_structure_data(existing, new):
    """Merge structure data, combining properties from both."""
    merged = existing.copy()

    # Merge properties from new data
    for key, value in new.items():
        if key in merged:
            # If both have the same key, prefer the new value if it's more detailed
            if isinstance(value, dict) and isinstance(merged[key], dict):
                # Recursively merge nested objects
                merged[key] = merge_structure_data(merged[key], value)
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Combine lists, removing duplicates
                merged[key] = list(set(merged[key] + value))
            else:
                # Prefer new value if it's not empty/null
                if value and value != "" and value != "unknown":
                    merged[key] = value
        else:
            merged[key] = value

    return merged


def merge_lot_data(existing, new):
    """Merge lot data, combining properties from both."""
    merged = existing.copy()

    # Merge properties from new data
    for key, value in new.items():
        if key in merged:
            # If both have the same key, prefer the new value if it's more detailed
            if isinstance(value, dict) and isinstance(merged[key], dict):
                # Recursively merge nested objects
                merged[key] = merge_lot_data(merged[key], value)
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Combine lists, removing duplicates
                merged[key] = list(set(merged[key] + value))
            else:
                # Prefer new value if it's not empty/null
                if value and value != "" and value != "unknown":
                    merged[key] = value
        else:
            merged[key] = value

    return merged


def merge_utility_data(existing, new):
    """Merge utility data, combining objects."""
    # Utility should be an object, not an array
    if isinstance(existing, dict) and isinstance(new, dict):
        # Merge dictionaries
        merged = existing.copy()
        for key, value in new.items():
            if key in merged:
                if isinstance(value, list) and isinstance(merged[key], list):
                    # Handle list merging with deduplication
                    combined = merged[key] + value
                    unique_items = []
                    seen_items = set()

                    for item in combined:
                        if isinstance(item, dict):
                            item_str = json.dumps(item, sort_keys=True)
                            if item_str not in seen_items:
                                unique_items.append(item)
                                seen_items.add(item_str)
                        else:
                            if item not in seen_items:
                                unique_items.append(item)
                                seen_items.add(item)

                    merged[key] = unique_items
                else:
                    # For non-list values, prefer non-null values
                    if value is not None:
                        merged[key] = value
            else:
                merged[key] = value
        return merged
    elif isinstance(existing, list) and isinstance(new, list):
        # If both are lists, convert to object format
        # Take the first item as the base object
        base_obj = existing[0] if existing else {}
        new_obj = new[0] if new else {}
        return merge_utility_data(base_obj, new_obj)
    elif isinstance(existing, list):
        # Convert existing list to object
        base_obj = existing[0] if existing else {}
        return merge_utility_data(base_obj, new)
    elif isinstance(new, list):
        # Convert new list to object
        new_obj = new[0] if new else {}
        return merge_utility_data(existing, new_obj)
    else:
        # If types don't match, return the new data
        return new


def merge_layout_data(existing, new):
    """Merge layout data, combining properties and appliances."""
    merged = existing.copy()

    # Clean both existing and new data by removing invalid fields
    def clean_layout_data(data):
        if not isinstance(data, dict):
            return data

        cleaned = data.copy()

        # Remove pool-related fields from layout
        pool_fields = ["pool_condition", "pool_equipment", "pool_type", "spa_type", "appliances", "pool_surface_type", "pool_water_quality"]
        for field in pool_fields:
            if field in cleaned:
                cleaned.pop(field, None)
                logger.debug(f"    [DEBUG] Removed pool field '{field}' from layout during merge")

        # Also remove any fields that don't belong in layout schema
        # Fetch layout schema from IPFS to validate fields
        layout_schema = fetch_schema_from_ipfs(IPFS_SCHEMA_CIDS["layout"])
        if layout_schema and isinstance(layout_schema, dict):
            layout_schema_properties = layout_schema.get("properties", {})
            if layout_schema_properties:
                for field in list(cleaned.keys()):
                    if field not in layout_schema_properties:
                        cleaned.pop(field, None)
                        logger.debug(f"    [DEBUG] Removed invalid field '{field}' from layout during merge")

        return cleaned

    # Clean the data before merging
    cleaned_existing = clean_layout_data(existing)
    cleaned_new = clean_layout_data(new)

    merged = cleaned_existing.copy()

    # Merge basic properties
    for key, value in cleaned_new.items():
        if key == "appliances":
            # Handle appliances separately
            continue
        elif key == "space_type":
            # Clean and normalize space_type
            if value and isinstance(value, str):
                cleaned_value = value.strip()
                if cleaned_value and cleaned_value != "unknown":
                    merged[key] = cleaned_value
        elif key in merged:
            if isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key] = merge_layout_data(merged[key], value)
            elif isinstance(value, list) and isinstance(merged[key], list):
                merged[key] = list(set(merged[key] + value))
            else:
                if value and value != "" and value != "unknown":
                    merged[key] = value
        else:
            merged[key] = value

    # Merge appliances
    existing_appliances = merged.get("appliances", [])
    new_appliances = new.get("appliances", [])

    if isinstance(existing_appliances, dict):
        existing_appliances = [existing_appliances]
    if isinstance(new_appliances, dict):
        new_appliances = [new_appliances]

    # Combine appliances, removing duplicates based on appliance_type
    all_appliances = existing_appliances + new_appliances
    unique_appliances = []
    seen_types = set()

    for appliance in all_appliances:
        if isinstance(appliance, dict):
            appliance_type = appliance.get("appliance_type", appliance.get("type", "unknown"))
            if appliance_type not in seen_types:
                unique_appliances.append(appliance)
                seen_types.add(appliance_type)

    merged["appliances"] = unique_appliances

    return merged


def merge_appliance_data(existing, new):
    """Merge appliance data, combining properties."""
    merged = existing.copy()

    # Merge properties from new data
    for key, value in new.items():
        if key in merged:
            if isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key] = merge_appliance_data(merged[key], value)
            elif isinstance(value, list) and isinstance(merged[key], list):
                merged[key] = list(set(merged[key] + value))
            else:
                if value and value != "" and value != "unknown":
                    merged[key] = value
        else:
            merged[key] = value

    return merged


def process_batch_worker(batch_info):
    """
    Worker function for processing a single batch in parallel.
    Returns (batch_number, result, cost, image_batch, success)
    """
    batch_number, image_batch, prompt, output_dir = batch_info

    try:
        logger.info(f"    [→] Processing batch {batch_number:02d} ({len(image_batch)} images) [PARALLEL]")
        result, cost = call_openai_optimized(image_batch, prompt)

        if result:
            # Save extracted output
            out_path = os.path.join(output_dir, f"batch_{batch_number:02d}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"    [✔] Saved: batch_{batch_number:02d}.json [PARALLEL]")
            return (batch_number, result, cost, image_batch, True)
        else:
            logger.error(f"    [✗] Failed: batch_{batch_number:02d} [PARALLEL]")
            return (batch_number, None, cost, image_batch, False)

    except Exception as e:
        logger.error(f"    [ERROR] Exception in batch {batch_number:02d}: {e}")
        return (batch_number, None, 0.0, image_batch, False)


def process_multiple_batches_parallel(image_files, prompt, output_dir, address, folder_path, start_batch_number=None):
    """Process images in multiple batches using parallel processing."""
    if start_batch_number is None:
        start_batch_number = get_next_batch_number(output_dir)

    total_batches = ceil(len(image_files) / 10)
    property_cost = 0.0
    property_cid = generate_clean_cid("property", address.replace("/", "_").replace(" ", "_"))
    all_relationships = []

    logger.info(f"    [→] Processing {total_batches} batches in PARALLEL starting from batch {start_batch_number:02d}")
    logger.info(f"    [→] Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")

    # Prepare batch information for parallel processing
    batch_jobs = []
    for i, image_batch in enumerate(chunk_list(sorted(image_files), 10)):
        batch_number = start_batch_number + i
        out_path = os.path.join(output_dir, f"batch_{batch_number:02d}.json")

        if os.path.exists(out_path):
            logger.info(f"    [✓] Skipping batch {batch_number:02d} (already exists)")
            continue

        batch_jobs.append((batch_number, image_batch, prompt, output_dir))

    # Process batches in parallel using ThreadPoolExecutor
    if batch_jobs:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            # Submit all jobs
            future_to_batch = {executor.submit(process_batch_worker, job): job[0] for job in batch_jobs}

            # Process completed results
            for future in future_to_batch:
                batch_number, result, cost, image_batch, success = future.result()
                property_cost += cost

                if success and result:
                    # Generate individual object files for this batch
                    logger.info(f"    [→] Generating individual object files for batch {batch_number:02d}...")
                    batch_object_files = generate_individual_object_files(result, image_batch, output_dir, batch_number)

                    # Generate relationships for this batch using clean CIDs
                    batch_relationships = generate_relationships_from_object_files(batch_object_files, image_batch,
                                                                                   property_cid, batch_number,
                                                                                   folder_path)

                    # Save batch relationships
                    relationships_path = os.path.join(output_dir, f"relationships_{batch_number:02d}.json")
                    with open(relationships_path, "w") as f:
                        json.dump(batch_relationships, f, indent=2)

                    batch_objects = (len(batch_object_files["layouts"]) +
                                     len(batch_object_files["appliances"]) +
                                     len(batch_object_files["property_objects"]))

                    logger.info(f"    [✔] Saved: relationships_{batch_number:02d}.json")
                    logger.info(f"    [📊] Batch {batch_number:02d}: {len(batch_relationships)} relationships | {batch_objects} object files")

                    all_relationships.extend(batch_relationships)

    # Save combined relationships for the entire property
    if all_relationships:
        combined_relationships_path = os.path.join(output_dir, "all_relationships.json")

        # If all_relationships.json already exists, load it and append new relationships
        existing_relationships = []
        if os.path.exists(combined_relationships_path):
            try:
                with open(combined_relationships_path, "r") as f:
                    existing_relationships = json.load(f)
                logger.info(f"    [→] Found existing relationships file with {len(existing_relationships)} relationships")
            except Exception as e:
                logger.warning(f"    [!] Warning: Could not load existing relationships: {e}")

        # Combine existing and new relationships
        combined_relationships = existing_relationships + all_relationships

        with open(combined_relationships_path, "w") as f:
            json.dump(combined_relationships, f, indent=2)

        logger.info(f"    [✔] Saved: all_relationships.json")
        logger.info(f"    [📊] FINAL: {len(combined_relationships)} total relationships ({len(all_relationships)} new)")

    return property_cost


def process_images_single_call_s3(image_objects, prompt, output_dir, property_id, schemas=None):
    """Process all S3 images in a single API call and group objects by space type."""
    logger.info(f"    [→] Processing all {len(image_objects)} images in single call")
    result, cost = call_openai_optimized_s3(image_objects, prompt)

    if result:
        # Save the original extracted output
        out_path = os.path.join(output_dir, "analysis_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"    [✔] Saved: analysis_result.json")

        # Generate grouped object files by space type
        logger.info(f"    [→] Generating grouped object files by space type...")
        grouped_files = generate_grouped_object_files_s3(result, image_objects, output_dir, property_id)

        # Generate relationships using clean CIDs and IPFS schema
        property_cid = generate_clean_cid("property", property_id.replace("/", "_").replace(" ", "_"))
        relationship_schema = schemas.get("relationship") if schemas else None
        relationships = generate_relationships_from_grouped_files_s3(grouped_files, image_objects, property_cid, property_id, relationship_schema)

        # Save relationships
        relationships_path = os.path.join(output_dir, "relationships.json")
        with open(relationships_path, "w") as f:
            json.dump(relationships, f, indent=2)

        logger.info(f"    [✔] Saved: relationships.json")
        logger.info(f"    [📊] Generated {len(relationships)} relationships for {len(image_objects)} images")

        # Print summary of created files
        total_objects = (len(grouped_files["layouts"]) +
                         len(grouped_files["appliances"]) +
                         len(grouped_files["property_objects"]) +
                         len(grouped_files["images"]))
        logger.info(f"    [📊] Created {total_objects} grouped object files")

    return cost


def process_images_in_single_call(image_files, prompt, output_dir, address, folder_path):
    """Process multiple images in a single API call and generate individual object files."""
    # Get the next available batch number to avoid overwriting
    start_batch_number = get_next_batch_number(output_dir)

    if len(image_files) <= 10:
        logger.info(f"    [→] Processing all {len(image_files)} images in single call (starting from batch {start_batch_number:02d})")
        result, cost = call_openai_optimized(image_files, prompt)

        if result:
            batch_number = start_batch_number

            # Save the original extracted output
            out_path = os.path.join(output_dir, f"batch_{batch_number:02d}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"    [✔] Saved: batch_{batch_number:02d}.json")

            # Generate individual object files (including image files)
            logger.info(f"    [→] Generating individual object files...")
            object_files = generate_individual_object_files(result, image_files, output_dir, batch_number)

            # Generate relationships using clean CIDs
            property_cid = generate_clean_cid("property", address.replace("/", "_").replace(" ", "_"))
            relationships = generate_relationships_from_object_files(object_files, image_files, property_cid,
                                                                     batch_number, folder_path)

            # Save relationships (just the list, no metadata wrapper)
            relationships_path = os.path.join(output_dir, f"relationships_{batch_number:02d}.json")
            with open(relationships_path, "w") as f:
                json.dump(relationships, f, indent=2)

            logger.info(f"    [✔] Saved: relationships_{batch_number:02d}.json")
            logger.info(f"    [📊] Generated {len(relationships)} relationships for {len(image_files)} images")

            # Update all_relationships.json file - APPEND, don't overwrite
            combined_relationships_path = os.path.join(output_dir, "all_relationships.json")

            # Load existing relationships if file exists
            existing_relationships = []
            if os.path.exists(combined_relationships_path):
                try:
                    with open(combined_relationships_path, "r") as f:
                        existing_relationships = json.load(f)
                    logger.info(f"    [→] Found existing relationships file with {len(existing_relationships)} relationships")
                except Exception as e:
                    logger.warning(f"    [!] Warning: Could not load existing relationships: {e}")

            # Combine existing and new relationships
            combined_relationships = existing_relationships + relationships

            # Save combined relationships
            with open(combined_relationships_path, "w") as f:
                json.dump(combined_relationships, f, indent=2)

            logger.info(f"    [✔] Updated: all_relationships.json")
            logger.info(f"    [📊] TOTAL: {len(combined_relationships)} total relationships ({len(relationships)} new)")

            # Print summary of created files
            total_objects = (len(object_files["layouts"]) +
                             len(object_files["appliances"]) +
                             len(object_files["property_objects"]) +
                             len(object_files["images"]))
            logger.info(f"    [📊] Created {total_objects} individual object files")

        return cost
    else:
        return process_multiple_batches_parallel(image_files, prompt, output_dir, address, folder_path,
                                                 start_batch_number)


def generate_grouped_object_files_s3(result, image_objects, output_dir, property_id):
    """
    Generate grouped object files by space type for S3 images.
    Groups layouts by space type, lots together, structures together.
    """
    grouped_files = {
        "layouts": {},
        "appliances": {},
        "property_objects": {},
        "images": {}
    }

    # Group layouts by space type
    layout_groups = {}
    if "layouts" in result:
        for layout in result["layouts"]:
            space_type = layout.get("space_type", "unknown")
            if space_type not in layout_groups:
                layout_groups[space_type] = []
            layout_groups[space_type].append(layout)

    # Save grouped layout files
    for space_type, layouts in layout_groups.items():
        if layouts:
            filename = f"layout_{space_type.lower().replace(' ', '_')}_photo_metadata.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(layouts, f, indent=2)
            grouped_files["layouts"][space_type] = filename
            logger.info(f"    [✔] Saved: {filename} ({len(layouts)} layouts)")

    # Group lots together
    if "lots" in result and result["lots"]:
        filename = "lot_photo_metadata.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(result["lots"], f, indent=2)
        grouped_files["property_objects"]["lot"] = filename
        logger.info(f"    [✔] Saved: {filename} ({len(result['lots'])} lots)")

    # Group structures together
    if "structures" in result and result["structures"]:
        filename = "structure_photo_metadata.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(result["structures"], f, indent=2)
        grouped_files["property_objects"]["structure"] = filename
        logger.info(f"    [✔] Saved: {filename} ({len(result['structures'])} structures)")

    # Group utilities together
    if "utilities" in result and result["utilities"]:
        filename = "utility_photo_metadata.json"
        filepath = os.path.join(output_dir, filename)

        # Ensure utility is saved as an object, not an array
        utility_data = result["utilities"]
        if isinstance(utility_data, list) and len(utility_data) > 0:
            # Convert array to object by taking the first item
            utility_data = utility_data[0]

        with open(filepath, "w") as f:
            json.dump(utility_data, f, indent=2)
        grouped_files["property_objects"]["utility"] = filename
        logger.info(f"    [✔] Saved: {filename} (utility object)")



    # Group appliances together
    if "appliances" in result and result["appliances"]:
        filename = "appliance_photo_metadata.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(result["appliances"], f, indent=2)
        grouped_files["appliances"]["all"] = filename
        logger.info(f"    [✔] Saved: {filename} ({len(result['appliances'])} appliances)")

    # Create image files (one per image)
    if "images" in result:
        for i, image_data in enumerate(result["images"]):
            # Get the original filename from image_objects
            if i < len(image_objects):
                original_filename = image_objects[i]["Key"].split("/")[-1]
                # Remove extension and create clean filename
                base_name = os.path.splitext(original_filename)[0]
                filename = f"{base_name}_photo_metadata.json"
            else:
                filename = f"image_{i:03d}_photo_metadata.json"

            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(image_data, f, indent=2)
            grouped_files["images"][filename.replace("_photo_metadata.json", "")] = filename

        logger.info(f"    [✔] Saved: {len(grouped_files['images'])} image files")

    return grouped_files


def generate_relationships_from_grouped_files_s3(grouped_files, image_objects, property_cid, property_id, relationship_schema=None):
    """
    Generate relationships using clean, meaningful CIDs based on grouped filenames for S3 images.
    Uses grouped files by space type instead of individual files.
    """
    relationships = []

    # Get image CIDs from the generated image files (these are already clean filenames)
    image_cids = grouped_files["images"]  # This is a dict: {image_cid -> filename}

    # Create property -> image relationships using image CIDs (which are filenames without .json)
    for image_cid in image_cids.keys():
        relationships.append(create_relationship(property_cid, image_cid, "property_has_file", relationship_schema))

    # Property-level object relationships using grouped filenames as CIDs
    for obj_type, filename in grouped_files["property_objects"].items():
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

    # Layout relationships using grouped filenames as CIDs (by space type)
    for space_type, filename in grouped_files["layouts"].items():
        # Use filename without .json extension as CID
        layout_cid = filename.replace(".json", "")
        relationships.append(create_relationship(property_cid, layout_cid, "property_has_layout", relationship_schema))

        # REMOVED: Layout to image relationships as requested
        # Link layout to all images in batch
        # for image_cid in image_cids.keys():
        #     relationships.append(create_relationship(layout_cid, image_cid, "layout_has_image", relationship_schema))

    # Appliance relationships using grouped filenames as CIDs
    for appliance_key, filename in grouped_files["appliances"].items():
        # Use filename without .json extension as CID
        appliance_cid = filename.replace(".json", "")
        relationships.append(create_relationship(property_cid, appliance_cid, "property_has_appliance", relationship_schema))

    return relationships


def process_s3_folder(folder_name, prompt, schemas=None):
    """Process a single S3 folder containing images in batches of 10."""
    start_time = time.time()

    # Extract property_id from the folder path (e.g., "52434205310037080/dining_room" -> "52434205310037080")
    if "/" in folder_name:
        property_id = folder_name.split("/")[0]
    else:
        property_id = folder_name

    output_dir = os.path.join(OUTPUT_BASE_FOLDER, property_id)
    os.makedirs(output_dir, exist_ok=True)

    # List all images in the S3 folder
    image_objects = list_s3_images_in_folder(folder_name)

    if not image_objects:
        logger.info(f"[-] No images found in S3 folder: {folder_name}")
        return 0.0

    # Check for existing batches and get next batch number
    next_batch = get_next_batch_number(output_dir)
    total_batches = ceil(len(image_objects) / 10)

    if next_batch > 1:
        logger.info(f"\n[+] Continuing S3 folder: {folder_name} | {len(image_objects)} images in {total_batches} batches (starting from batch {next_batch:02d})")
        logger.info(f"[+] Found existing batches 1-{next_batch - 1:02d}")
    else:
        logger.info(f"\n[+] Processing S3 folder: {folder_name} | {len(image_objects)} images in {total_batches} batches")

    logger.info(f"[+] S3 folder: {S3_BASE_PREFIX}{folder_name}/")
    logger.info(f"[+] Output directory: {output_dir}")

    # Process the images in batches
    property_cost = 0.0
    all_relationships = []

    try:
        for batch_idx, batch_start in enumerate(range(0, len(image_objects), 10), start=1):
            batch_images = image_objects[batch_start:batch_start+10]
            logger.info(f"    [→] Processing batch {batch_idx:02d} ({len(batch_images)} images)")

            # Add timeout and error handling
            try:
                logger.debug(f"    [DEBUG] Starting OpenAI API call for batch {batch_idx:02d}")
                import signal

                # Set a timeout for the entire batch processing
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Batch {batch_idx:02d} processing timed out")

                # Set 5 minute timeout for each batch
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)  # 5 minutes

                try:
                    result, cost = call_openai_optimized_s3(batch_images, prompt)
                    signal.alarm(0)  # Cancel the alarm
                    logger.debug(f"    [DEBUG] OpenAI API call completed for batch {batch_idx:02d}")
                    property_cost += cost
                except TimeoutError as te:
                    signal.alarm(0)  # Cancel the alarm
                    logger.info(f"    [!] Timeout for batch {batch_idx:02d}: {te}")
                    continue

                if result:
                    logger.debug(f"    [DEBUG] Processing result for batch {batch_idx:02d}")
                    # Save batch result
                    batch_filename = f"batch_{batch_idx:02d}.json"
                    batch_path = os.path.join(output_dir, batch_filename)
                    with open(batch_path, "w") as f:
                        json.dump(result, f, indent=2)
                    logger.info(f"    [✔] Saved: {batch_filename}")

                    # Generate or update object files for this batch (merge instead of create new)
                    logger.debug(f"    [DEBUG] Generating/updating object files for batch {batch_idx:02d}")
                    object_files = merge_and_update_object_files_s3(result, batch_images, output_dir, batch_idx)

                    # Generate relationships for this batch
                    logger.debug(f"    [DEBUG] Generating relationships for batch {batch_idx:02d}")
                    property_cid = generate_clean_cid("property", property_id.replace("/", "_").replace(" ", "_"))
                    relationship_schema = schemas.get("relationship") if schemas else None
                    relationships = generate_individual_relationship_files_s3(object_files, batch_images, property_cid, property_id, relationship_schema, output_dir)

                    # Create main relationship file
                    create_main_relationship_file(relationships, output_dir, property_id)

                    logger.debug(f"    [DEBUG] Completed processing batch {batch_idx:02d}")
                else:
                    logger.info(f"    [!] No result for batch {batch_idx:02d}")

            except Exception as e:
                logger.error(f"    [!] Error processing batch {batch_idx:02d}: {e}")
                import traceback
                traceback.print_exc()
                continue

    except KeyboardInterrupt:
        logger.info(f"    [!] Processing interrupted for {folder_name}")
        return property_cost
    except Exception as e:
        logger.error(f"    [!] Fatal error processing {folder_name}: {e}")
        return property_cost

    # Save combined relationships for the property
    if all_relationships:
        combined_relationships_path = os.path.join(output_dir, "all_relationships.json")

        # Load existing relationships if file exists
        existing_relationships = []
        if os.path.exists(combined_relationships_path):
            try:
                with open(combined_relationships_path, "r") as f:
                    existing_relationships = json.load(f)
                logger.info(f"    [→] Found existing relationships file with {len(existing_relationships)} relationships")
            except Exception as e:
                logger.warning(f"    [!] Warning: Could not load existing relationships: {e}")

        # Combine existing and new relationships, avoiding duplicates
        combined_relationships = existing_relationships + all_relationships

        # Remove duplicate relationships based on from, to, and type
        unique_relationships = []
        seen_relationships = set()

        for rel in combined_relationships:
            if isinstance(rel, dict) and "properties" in rel:
                props = rel["properties"]
                rel_key = (props.get("from"), props.get("to"), props.get("type"))
                if rel_key not in seen_relationships:
                    unique_relationships.append(rel)
                    seen_relationships.add(rel_key)

        with open(combined_relationships_path, "w") as f:
            json.dump(unique_relationships, f, indent=2)
        logger.info(f"    [✔] Saved: all_relationships.json ({len(unique_relationships)} unique relationships)")

    elapsed = time.time() - start_time
    logger.info(f"[✓] Done: {folder_name} | 💰 ${property_cost:.4f} | ⏱️ {elapsed:.1f} sec")
    return property_cost


def process_property_row(address, folder_path, prompt, executor=None):
    start_time = time.time()
    if not os.path.exists(folder_path):
        logger.info(f"[!] Folder not found: {folder_path}")
        return 0.0

    output_dir = os.path.join(OUTPUT_BASE_FOLDER, address.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        logger.info(f"[-] No images in {folder_path}")
        return 0.0

    # Check for existing batches and get next batch number
    next_batch = get_next_batch_number(output_dir)
    total_batches = ceil(len(image_files) / 10)

    if next_batch > 1:
        logger.info(f"\n[+] Continuing property: {address} | {len(image_files)} images in {total_batches} batches (starting from batch {next_batch:02d})")
        logger.info(f"[+] Found existing batches 1-{next_batch - 1:02d}")
    else:
        logger.info(f"\n[+] Processing property: {address} | {len(image_files)} images in {total_batches} batches")

    logger.info(f"[+] Folder path: {folder_path}")

    # Pass folder_path to the processing function
    property_cost = process_images_in_single_call(image_files, prompt, output_dir, address, folder_path)

    elapsed = time.time() - start_time
    logger.info(f"[✓] Done: {address} | 💰 ${property_cost:.4f} | ⏱️ {elapsed:.1f} sec")
    return property_cost


def print_final_statistics():
    """Print comprehensive final statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL PROCESSING STATISTICS")
    logger.info("=" * 60)
    logger.info(f"🖼️  Total Images Processed: {TOTAL_IMAGES_PROCESSED:,}")
    logger.info(f"🔤 Total Prompt Tokens: {TOTAL_PROMPT_TOKENS:,}")
    logger.info(f"🔤 Total Completion Tokens: {TOTAL_COMPLETION_TOKENS:,}")
    logger.info(f"🔤 Total Tokens: {TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS:,}")
    logger.info(f"💰 Total Cost: ${TOTAL_COST:.4f}")

    if TOTAL_IMAGES_PROCESSED > 0:
        logger.info(f"📈 Average Cost per Image: ${TOTAL_COST / TOTAL_IMAGES_PROCESSED:.4f}")
        logger.info(f"📈 Average Tokens per Image: {(TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS) / TOTAL_IMAGES_PROCESSED:.1f}")
    else:
        logger.info("📈 Average Cost per Image: N/A (no images processed)")
        logger.info("📈 Average Tokens per Image: N/A (no images processed)")

    if TOTAL_IMAGES_PROCESSED > 0:
        original_size_estimate = TOTAL_IMAGES_PROCESSED * 2048 * 2048 * 3
        optimized_size_estimate = TOTAL_IMAGES_PROCESSED * 1024 * 1024 * 3
        size_reduction_percent = ((original_size_estimate - optimized_size_estimate) / original_size_estimate) * 100
        logger.info(f"🗜️  Estimated Image Size Reduction: {size_reduction_percent:.1f}%")
    else:
        logger.info("🗜️  Estimated Image Size Reduction: N/A (no images processed)")

    logger.info(f"🗜️  Max Image Resolution: {MAX_IMAGE_SIZE[0]}x{MAX_IMAGE_SIZE[1]} (JPEG Quality: {JPEG_QUALITY}%)")
    logger.info("=" * 60)


def process_all_local_properties_from_folders(property_folders, prompt, schemas=None, batch_size=10, max_workers=12, parallel_categories=False, category_workers=12):
    """Process all local properties from discovered folders"""
    logger.info(f"🖥️  Processing {len(property_folders)} local properties...")

    total_successful = 0
    total_failed = 0

    for property_id in property_folders:
        logger.info(f"\n{'='*80}")
        logger.info(f"🏠 Processing Local Property: {property_id}")
        logger.info(f"{'='*80}")

        # Check if property has categorized folders
        property_path = os.path.join("images", property_id)
        if not os.path.exists(property_path):
            logger.warning(f"⚠️  Property folder {property_path} not found, skipping...")
            total_failed += 1
            continue

        # Get all category folders for this property
        categories = []
        for item in os.listdir(property_path):
            item_path = os.path.join(property_path, item)
            if os.path.isdir(item_path):
                categories.append(item)

        if not categories:
            logger.warning(f"⚠️  No category folders found for property {property_id}, skipping...")
            total_failed += 1
            continue

        logger.info(f"📁 Found {len(categories)} category folders for {property_id}: {', '.join(categories)}")

        # Process categories based on parallel flag
        if parallel_categories:
            logger.info(f"🚀 Processing categories in parallel for {property_id}")
            result = process_local_categories_parallel(
                property_id, categories, prompt, schemas,
                batch_size, max_workers, category_workers
            )
            property_success = result['success']
            if property_success:
                logger.info(f"✅ Successfully processed property {property_id} with parallel categories")
            else:
                logger.warning(f"⚠️  Failed to process property {property_id} with parallel categories")
        else:
            # Process each category folder sequentially
            logger.info(f"🔄 Processing categories sequentially for {property_id}")
            property_success = False
            for category in categories:
                logger.info(f"\n{'='*60}")
                logger.info(f"🏠 Processing Category: {category}")
                logger.info(f"{'='*60}")

                try:
                    # Process the category folder
                    success = process_local_category_folder(property_id, category, prompt, schemas, batch_size, max_workers)
                    if success:
                        property_success = True
                        logger.info(f"✅ Successfully processed category {category}")
                    else:
                        logger.warning(f"⚠️  Failed to process category {category}")
                except Exception as e:
                    logger.error(f"❌ Error processing category {category}: {e}")

            if property_success:
                logger.info(f"✅ Successfully processed property {property_id}")
            else:
                logger.warning(f"⚠️  Failed to process property {property_id}")

        if property_success:
            total_successful += 1
        else:
            total_failed += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"LOCAL PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total properties processed: {len(property_folders)}")
    logger.info(f"Successful: {total_successful}")
    logger.info(f"Failed: {total_failed}")

    return total_successful > 0


def process_local_categories_parallel(property_id, categories, prompt, schemas=None, batch_size=10, max_workers=12, category_workers=12):
    """Process local categories in parallel."""
    logger.info(f"🚀 Using parallel category processing with {category_workers} workers")
    logger.info(f"📁 Found {len(categories)} category folders for {property_id}: {', '.join(categories)}")

    total_cost = 0
    completed_categories = []
    failed_categories = []

    def process_category_worker(category):
        """Worker function to process a single category."""
        try:
            logger.info(f"🚀 Processing category {category} for {property_id}")
            start_time = time.time()

            # Process the category
            result = process_local_category_folder(
                property_id, category, prompt, schemas,
                batch_size, max_workers
            )

            end_time = time.time()
            duration = end_time - start_time

            if result and result.get('success'):
                cost = result.get('cost', 0)
                total_cost += cost
                logger.info(f"✅ Completed category {category} (Cost: ${cost:.4f}, Duration: {duration:.1f}s)")
                return {'category': category, 'success': True, 'cost': cost, 'duration': duration}
            else:
                logger.error(f"❌ Failed to process category {category}")
                return {'category': category, 'success': False, 'error': 'Processing failed'}

        except Exception as e:
            logger.error(f"❌ Error processing category {category}: {e}")
            return {'category': category, 'success': False, 'error': str(e)}

    # Process categories in parallel
    with ThreadPoolExecutor(max_workers=category_workers) as executor:
        # Submit all category processing tasks
        future_to_category = {
            executor.submit(process_category_worker, category): category
            for category in categories
        }

        # Process completed tasks
        for future in as_completed(future_to_category):
            category = future_to_category[future]
            try:
                result = future.result(timeout=600)  # 10 minute timeout per category
                if result['success']:
                    completed_categories.append(result)
                else:
                    failed_categories.append(result)
            except Exception as e:
                logger.error(f"❌ Category {category} timed out or failed: {e}")
                failed_categories.append({'category': category, 'success': False, 'error': str(e)})

    # Summary
    logger.info(f"✅ Completed all categories for {property_id}")
    logger.info(f"📊 Results: {len(completed_categories)} successful, {len(failed_categories)} failed")
    logger.info(f"💰 Total cost: ${total_cost:.4f}")

    return {
        'success': len(failed_categories) == 0,
        'completed': completed_categories,
        'failed': failed_categories,
        'total_cost': total_cost
    }

def process_all_local_properties(seed_data_path, prompt, schemas=None, batch_size=3, max_workers=3):
    """Process all properties from local categorized folders"""
    try:
        # Load seed data
        df = pd.read_csv(seed_data_path)
        logger.info(f"✓ Loaded {len(df)} records from seed data CSV")

        # Get all property IDs
        property_ids = df['parcel_id'].astype(str).tolist()
        logger.info(f"📁 Processing {len(property_ids)} properties from local folders")

        total_processed = 0
        total_images = 0

        for property_id in property_ids:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing Property: {property_id}")
            logger.info(f"{'='*80}")

            # Check if local property folder exists
            local_property_path = os.path.join("images", property_id)
            if not os.path.exists(local_property_path):
                logger.warning(f"⚠️  Local property folder not found: {local_property_path}")
                continue

            # Get all category folders
            category_folders = []
            for item in os.listdir(local_property_path):
                item_path = os.path.join(local_property_path, item)
                if os.path.isdir(item_path):
                    category_folders.append(item)

            if not category_folders:
                logger.warning(f"⚠️  No category folders found for property {property_id}")
                continue

            logger.info(f"📁 Found {len(category_folders)} category folders for property {property_id}")

            # Process each category folder
            for category in category_folders:
                logger.info(f"\n🖼️  Processing category: {category}")
                success = process_local_category_folder(
                    property_id, category, prompt, schemas, batch_size, max_workers
                )
                if success:
                    total_processed += 1
                    # Count images in this category
                    category_path = os.path.join(local_property_path, category)
                    image_count = len([f for f in os.listdir(category_path)
                                     if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}])
                    total_images += image_count
                    logger.info(f"✅ Successfully processed {image_count} images in {category}")
                else:
                    logger.warning(f"⚠️  Failed to process category {category}")

        logger.info(f"\n🎉 Processing completed!")
        logger.info(f"📊 Total properties processed: {total_processed}")
        logger.info(f"📊 Total images processed: {total_images}")

        return total_processed > 0

    except Exception as e:
        logger.error(f"❌ Error processing local properties: {e}")
        return False

def main():
    import argparse
    import sys
    import pandas as pd

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Image Analysis for Real Estate Properties')
    parser.add_argument('--property-id', type=str, help='Specific property ID to process (optional)')
    parser.add_argument('--all-properties', action='store_true', help='Process all properties from seed.csv')
    parser.add_argument('--local-folders', action='store_true', help='Process from local categorized folders')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing (default: 10 - OpenAI limit)')
    parser.add_argument('--max-workers', type=int, default=12, help='Maximum workers for parallel processing (default: 12 for high performance)')
    parser.add_argument('--parallel-categories', action='store_true', help='Process categories in parallel instead of sequentially')
    parser.add_argument('--category-workers', type=int, default=12, help='Number of category workers for parallel processing (default: 12)')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory (default: output)')

    args = parser.parse_args()

    total_start = time.time()

    logger.info("🚀 Starting optimized real estate image processing with S3 and IPFS integration...")
    logger.info(f"🖼️  Image optimization: Max size {MAX_IMAGE_SIZE[0]}x{MAX_IMAGE_SIZE[1]}, JPEG quality {JPEG_QUALITY}%")
    logger.info(f"🚀 Single call processing: All images processed in one API call per folder")
    logger.info(f"☁️  S3 Bucket: {S3_BUCKET_NAME}")
    logger.info(f"🌐 IPFS Schemas: {len(IPFS_SCHEMA_CIDS)} schemas + 1 relationship schema")
    logger.info(f"📁 Output Structure: All files go directly to output/property_id/ (no subfolders)")
    logger.info(f"🔄 Data Merging: Updates existing files instead of creating new ones")
    logger.info(f"🔗 Individual Relationships: Creates separate relationship files with IPFS format")

    # Load schemas from IPFS
    logger.info(f"\n[→] Loading schemas from IPFS...")
    schemas = load_schemas_from_ipfs()

    if not schemas:
        logger.error("❌ Failed to load schemas from IPFS. Exiting.")
        return

    logger.info(f"✓ Successfully loaded {len(schemas)} schemas from IPFS")

    # Debug: Check which schemas were loaded
    for schema_name, schema_data in schemas.items():
        if schema_data:
            logger.info(f"✓ Loaded {schema_name} schema with {len(schema_data.get('properties', {}))} properties")
            # Log a sample of the schema structure
            if schema_data.get('properties'):
                sample_props = list(schema_data['properties'].keys())[:3]
                logger.info(f"  Sample properties: {sample_props}")
        else:
            logger.warning(f"⚠️  {schema_name} schema is empty")

    # Authenticate with AWS
    if not authenticate_aws():
        logger.error("❌ Failed to authenticate with AWS. Exiting.")
        return

    # Ensure bucket exists
    logger.info(f"\n[→] Ensuring S3 bucket {S3_BUCKET_NAME} exists...")
    try:
        from bucket_manager import BucketManager
        bucket_manager = BucketManager()
        bucket_manager.authenticate_aws()
        bucket_success = bucket_manager.ensure_bucket_exists(S3_BUCKET_NAME)
        if not bucket_success:
            logger.error("❌ Bucket setup failed! Cannot proceed.")
            return
        logger.info("✓ Bucket is ready!")
    except ImportError:
        logger.warning("⚠️  Bucket manager not available, skipping bucket check...")
        logger.info("✓ Continuing without bucket verification...")

    # Get properties based on arguments
    if args.local_folders:
        # Process from local categorized folders
        logger.info("🖥️  Processing from local categorized folders...")

        # Discover properties from images directory
        images_dir = "images"
        if not os.path.exists(images_dir):
            logger.error(f"❌ Images directory '{images_dir}' not found!")
            logger.error("Please ensure the images directory exists with property folders")
            return

        try:
            property_folders = []
            for item in os.listdir(images_dir):
                item_path = os.path.join(images_dir, item)
                if os.path.isdir(item_path):
                    property_folders.append(item)

            if not property_folders:
                logger.error(f"❌ No property folders found in {images_dir}!")
                logger.error("Please ensure the images directory contains property folders")
                return

            logger.info(f"✓ Found {len(property_folders)} property folders: {', '.join(property_folders)}")

        except Exception as e:
            logger.error(f"❌ Error discovering properties: {e}")
            return

        # For local folders, we'll load county data per property in the processing function
        # Create a base prompt without county data (will be enhanced per property)
        prompt = load_optimized_json_schema_prompt(None, schemas)
        success = process_all_local_properties_from_folders(
            property_folders, prompt, schemas,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            parallel_categories=args.parallel_categories,
            category_workers=args.category_workers
        )
        if success:
            logger.info("🎉 Local folder processing completed successfully!")
        else:
            logger.error("❌ Local folder processing failed!")
        return
    else:
        # Load properties from seed.csv for S3 processing
        seed_data_path = "seed.csv"

        if not os.path.exists(seed_data_path):
            logger.error(f"❌ {seed_data_path} not found!")
            logger.error("Please provide seed.csv with parcel_id,Address columns")
            return

        logger.info(f"✓ Found {seed_data_path}")

        # Load seed data to get property IDs
        try:
            df = pd.read_csv(seed_data_path)
            logger.info(f"✓ Loaded {len(df)} records from seed data CSV")

            properties = []
            for _, row in df.iterrows():
                parcel_id = str(row['parcel_id'])
                properties.append(parcel_id)

            logger.info(f"✓ Created {len(properties)} property mappings from seed.csv")

        except Exception as e:
            logger.error(f"Error loading seed data CSV: {e}")
            return

        if not properties:
            logger.error("❌ No properties found in seed.csv. Please ensure the file contains parcel_id and Address columns.")
            return

        logger.info(f"📁 Found {len(properties)} properties from seed.csv: {', '.join(properties)}")

        # Filter properties based on arguments
        if args.property_id:
            if args.property_id not in properties:
                logger.error(f"❌ Property {args.property_id} not found in seed.csv.")
                return
            properties_to_process = [args.property_id]
        elif args.all_properties:
            properties_to_process = properties
        else:
            logger.error("❌ Please specify either --property-id, --all-properties, or --local-folders")
            return

    logger.info(f"🎯 Processing {len(properties_to_process)} properties from S3: {', '.join(properties_to_process)}")

    # visual_tags removed - no longer needed

    # Process each property from S3
    total_cost = 0.0

    # Set a timeout for the entire processing (30 minutes)
    import signal
    def timeout_handler(signum, frame):
        logger.error("⏰ Processing timeout reached (30 minutes). Stopping...")
        raise TimeoutError("Processing timeout")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1800)  # 30 minutes timeout

    try:
        for property_id in properties_to_process:
            logger.info(f"\n{'='*80}")
            logger.info(f"🏠 Processing Property: {property_id}")
            logger.info(f"{'='*80}")

            # List all category folders for this property
            categories = list_s3_subfolders_for_property(property_id)

            if not categories:
                logger.warning(f"⚠️  No category folders found for property {property_id}, skipping...")
                continue

            logger.info(f"📁 Found {len(categories)} category folders for {property_id}: {', '.join(categories)}")

            # Load county layout data for this property
            county_layouts = load_county_layout_data(property_id)
            if county_layouts:
                logger.info(f"📋 Loaded {len(county_layouts)} county layout records for property {property_id}")
            else:
                logger.info(f"📋 No county layout data found for property {property_id}")

            # Process each category folder
            property_cost = 0.0
            for category in categories:
                # Create category-specific prompt with categorization instructions, IPFS schemas, and county data
                prompt = load_optimized_json_schema_prompt(category, schemas, county_layouts)

                logger.info(f"\n{'='*60}")
                logger.info(f"🏠 Processing Category: {category}")
                logger.info(f"{'='*60}")

                cost = process_s3_subfolder_multi_threaded(property_id, category, prompt, schemas,
                                                          batch_size=args.batch_size, max_workers=args.max_workers)
                property_cost += cost

            total_cost += property_cost
            logger.info(f"💰 Property {property_id} cost: ${property_cost:.4f}")

    except TimeoutError:
        logger.error("⏰ Processing timed out. Some properties may not have been processed.")
    except Exception as e:
        logger.error(f"❌ Unexpected error during processing: {e}")
    finally:
        signal.alarm(0)  # Cancel the alarm

    total_elapsed = time.time() - total_start
    logger.info(f"\n✅ All properties processed in {total_elapsed:.1f} seconds")
    logger.info(f"💰 Total cost: ${total_cost:.4f}")

    print_final_statistics()


def process_s3_folder_no_batching(folder_name, prompt, schemas=None):
    """Process a single S3 folder containing images without batching."""
    start_time = time.time()

    # Extract property_id from the folder path (e.g., "52434205310037080/dining_room" -> "52434205310037080")
    if "/" in folder_name:
        property_id = folder_name.split("/")[0]
    else:
        property_id = folder_name

    output_dir = os.path.join(OUTPUT_BASE_FOLDER, property_id)
    os.makedirs(output_dir, exist_ok=True)

    # List all images in the S3 folder
    image_objects = list_s3_images_in_folder(folder_name)

    if not image_objects:
        logger.info(f"[-] No images found in S3 folder: {folder_name}")
        return 0.0

    logger.info(f"\n[+] Processing S3 folder: {folder_name} | {len(image_objects)} images")
    logger.info(f"[+] S3 folder: {S3_BASE_PREFIX}{folder_name}/")
    logger.info(f"[+] Output directory: {output_dir}")

    # Process all images in a single call
    property_cost = 0.0

    try:
        logger.info(f"    [→] Processing all {len(image_objects)} images in single call")

        # Add timeout and error handling
        try:
            logger.debug(f"    [DEBUG] Starting OpenAI API call for all images")
            import signal

            # Set a timeout for the entire processing
            def timeout_handler(signum, frame):
                raise TimeoutError("Processing timed out")

            # Set 10 minute timeout for all images
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)  # 10 minutes

            try:
                result, cost = call_openai_optimized_s3(image_objects, prompt)
                signal.alarm(0)  # Cancel the alarm
                logger.debug(f"    [DEBUG] OpenAI API call completed")
                property_cost += cost
            except TimeoutError as te:
                signal.alarm(0)  # Cancel the alarm
                logger.info(f"    [!] Timeout: {te}")
                return property_cost

            if result:
                logger.debug(f"    [DEBUG] Processing result")
                # Save batch result
                batch_filename = f"analysis_result.json"
                batch_path = os.path.join(output_dir, batch_filename)
                with open(batch_path, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"    [✔] Saved: {batch_filename}")

                # Generate or update object files
                logger.debug(f"    [DEBUG] Generating/updating object files")
                object_files = merge_and_update_object_files_s3(result, image_objects, output_dir, 1)

                # Skip property.json creation as requested

                # Generate individual relationship files
                logger.debug(f"    [DEBUG] Generating individual relationship files")
                property_cid = generate_clean_cid("property", property_id.replace("/", "_").replace(" ", "_"))
                relationship_schema = schemas.get("relationship") if schemas else None
                relationships = generate_individual_relationship_files_s3(object_files, image_objects, property_cid, property_id, relationship_schema, output_dir)

                # Create main relationship file using IPFS schema
                logger.debug(f"    [DEBUG] Creating main relationship file")
                create_main_relationship_file(relationships, output_dir, property_id)

                logger.debug(f"    [DEBUG] Completed processing")
            else:
                logger.info(f"    [!] No result for processing")

        except Exception as e:
            logger.error(f"    [!] Error processing: {e}")
            import traceback
            traceback.print_exc()

    except KeyboardInterrupt:
        logger.info(f"    [!] Processing interrupted for {folder_name}")
        return property_cost
    except Exception as e:
        logger.error(f"    [!] Fatal error processing {folder_name}: {e}")
        return property_cost

    elapsed = time.time() - start_time
    logger.info(f"[✓] Done: {folder_name} | 💰 ${property_cost:.4f} | ⏱️ {elapsed:.1f} sec")
    return property_cost


def generate_individual_relationship_files_s3(object_files, image_objects, property_cid, property_id, relationship_schema=None, output_dir=None):
    """
    Generate simplified relationship files for speed.
    Returns a list of relationship file info for the main relationship file.
    """
    relationship_files = []

    # Skip IPFS schema fetching for speed - use simplified structure
    relationship_schema = None

    # Get image CIDs from the generated image files
    image_cids = object_files.get("images", {})  # This is a dict: {image_cid -> filename}

    # Create property -> file relationships for images (simplified)
    for image_cid, filename in image_cids.items():
        rel_filename = f"relationship_property_file_{filename}"
        rel_path = os.path.join(output_dir, rel_filename)

        # Relationship structure with property.json reference
        relationship_data = {
            "from": {"/": "./property.json"},
            "to": {"/": f"./{filename}"}
        }

        with open(rel_path, "w") as f:
            json.dump(relationship_data, f, indent=2)

        relationship_files.append({
            "filename": rel_filename,
            "type": "property_has_file"
        })

    # Property-level object relationships (simplified)
    for obj_type, filename in object_files.get("property_objects", {}).items():
        rel_filename = f"relationship_property_{obj_type}.json"
        rel_path = os.path.join(output_dir, rel_filename)

        # Relationship structure with property.json reference
        relationship_data = {
            "from": {"/": "./property.json"},
            "to": {"/": f"./{filename}"}
        }

        with open(rel_path, "w") as f:
            json.dump(relationship_data, f, indent=2)

        relationship_files.append({
            "filename": rel_filename,
            "type": f"property_has_{obj_type}"
        })

    # Layout relationships (simplified)
    for layout_key, filename in object_files.get("layouts", {}).items():
        rel_filename = f"relationship_property_layout_{layout_key}.json"
        rel_path = os.path.join(output_dir, rel_filename)

        # Relationship structure with property.json reference
        relationship_data = {
            "from": {"/": "./property.json"},
            "to": {"/": f"./{filename}"}
        }

        with open(rel_path, "w") as f:
            json.dump(relationship_data, f, indent=2)

        relationship_files.append({
            "filename": rel_filename,
            "type": "property_has_layout"
        })

    # Appliance relationships (simplified)
    for appliance_key, filename in object_files.get("appliances", {}).items():
        rel_filename = f"relationship_property_appliance_{appliance_key}.json"
        rel_path = os.path.join(output_dir, rel_filename)

        # Relationship structure with property.json reference
        relationship_data = {
            "from": {"/": "./property.json"},
            "to": {"/": f"./{filename}"}
        }

        with open(rel_path, "w") as f:
            json.dump(relationship_data, f, indent=2)

        relationship_files.append({
            "filename": rel_filename,
            "type": "property_has_appliance"
        })

    return relationship_files


def create_main_relationship_file(relationship_files, output_dir, property_id):
    """Create the main relationship file using IPFS schema format."""
    # Use the relationship schema CID as the filename
    main_relationship_file = f"{RELATIONSHIP_SCHEMA_CID}.json"
    filepath = os.path.join(output_dir, main_relationship_file)

    # Load existing relationships if file exists, otherwise create new
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                main_relationships = json.load(f)
            logger.info(f"    [→] Found existing main relationship file, merging...")
        except Exception as e:
            logger.error(f"    [!] Error reading existing main relationship file: {e}")
            main_relationships = {
                "label": "Photo Metadata",
                "relationships": {}
            }
    else:
        # Create new structure
        main_relationships = {
            "label": "Photo Metadata",
            "relationships": {}
        }

    # Ensure label and relationships are always properly set
    if "label" not in main_relationships or not main_relationships["label"]:
        main_relationships["label"] = "Photo Metadata"
    if "relationships" not in main_relationships:
        main_relationships["relationships"] = {}

    # Group relationships by type
    for relationship_file in relationship_files:
        relationship_type = relationship_file.get("type", "unknown")

        # Initialize the relationship type array if it doesn't exist
        if relationship_type not in main_relationships["relationships"]:
            main_relationships["relationships"][relationship_type] = []

        # Add the relationship entry with simple reference
        relationship_entry = {
            "/": f"./{relationship_file['filename']}"
        }

        # Check if this relationship already exists to avoid duplicates
        existing_filenames = [rel.get("/", "").replace("./", "") for rel in main_relationships["relationships"][relationship_type]]
        if relationship_file['filename'] not in existing_filenames:
            main_relationships["relationships"][relationship_type].append(relationship_entry)

    # Save the main relationship file
    with open(filepath, "w") as f:
        json.dump(main_relationships, f, indent=2)

    logger.info(f"    [✔] Updated: {main_relationship_file}")
    return main_relationship_file


def process_s3_property_no_batching(property_id, prompt, schemas=None):
    """Process an entire S3 property folder containing images from all subfolders."""
    start_time = time.time()

    output_dir = os.path.join(OUTPUT_BASE_FOLDER, property_id)
    os.makedirs(output_dir, exist_ok=True)

    # List all images in the entire property folder (including all subfolders)
    image_objects = list_s3_images_in_property(property_id)

    if not image_objects:
        logger.info(f"[-] No images found in S3 property: {property_id}")
        return 0.0

    logger.info(f"\n[+] Processing S3 property: {property_id} | {len(image_objects)} images from all subfolders")
    logger.info(f"[+] S3 property: {S3_BASE_PREFIX}{property_id}/")
    logger.info(f"[+] Output directory: {output_dir}")

    # Process all images in a single call
    property_cost = 0.0

    try:
        logger.info(f"    [→] Processing all {len(image_objects)} images in single call")

        # Add timeout and error handling
        try:
            logger.debug(f"    [DEBUG] Starting OpenAI API call for all images")
            import signal

            # Set a timeout for the entire processing
            def timeout_handler(signum, frame):
                raise TimeoutError("Processing timed out")

            # Set 10 minute timeout for all images
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)  # 10 minutes

            try:
                result, cost = call_openai_optimized_s3(image_objects, prompt)
                signal.alarm(0)  # Cancel the alarm
                logger.debug(f"    [DEBUG] OpenAI API call completed")
                property_cost += cost
            except TimeoutError as te:
                signal.alarm(0)  # Cancel the alarm
                logger.info(f"    [!] Timeout: {te}")
                return property_cost

            if result:
                logger.debug(f"    [DEBUG] Processing result")
                # Save batch result
                batch_filename = f"analysis_result.json"
                batch_path = os.path.join(output_dir, batch_filename)
                with open(batch_path, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"    [✔] Saved: {batch_filename}")

                # Generate or update object files
                logger.debug(f"    [DEBUG] Generating/updating object files")
                object_files = merge_and_update_object_files_s3(result, image_objects, output_dir, 1)

                # Create property.json file
                logger.debug(f"    [DEBUG] Creating property.json file")
                property_data = {
                    "parcel_id": property_id
                }
                property_path = os.path.join(output_dir, "property.json")
                with open(property_path, "w") as f:
                    json.dump(property_data, f, indent=2)
                logger.info(f"    [✔] Saved: property.json")

                # Generate individual relationship files
                logger.debug(f"    [DEBUG] Generating individual relationship files")
                property_cid = generate_clean_cid("property", property_id.replace("/", "_").replace(" ", "_"))
                relationship_schema = schemas.get("relationship") if schemas else None
                relationships = generate_individual_relationship_files_s3(object_files, image_objects, property_cid, property_id, relationship_schema, output_dir)

                # Create main relationship file using IPFS schema
                logger.debug(f"    [DEBUG] Creating main relationship file")
                create_main_relationship_file(relationships, output_dir, property_id)

                logger.debug(f"    [DEBUG] Completed processing")
            else:
                logger.info(f"    [!] No result for processing")

        except Exception as e:
            logger.error(f"    [!] Error processing: {e}")
            import traceback
            traceback.print_exc()

    except KeyboardInterrupt:
        logger.info(f"    [!] Processing interrupted for {property_id}")
        return property_cost
    except Exception as e:
        logger.error(f"    [!] Fatal error processing {property_id}: {e}")
        return property_cost

    elapsed = time.time() - start_time
    logger.info(f"[✓] Done: {property_id} | 💰 ${property_cost:.4f} | ⏱️ {elapsed:.1f} sec")
    return property_cost


def process_s3_subfolder_no_batching(property_id, subfolder, prompt, schemas=None):
    """Process a single S3 subfolder and merge output with existing property folder."""
    start_time = time.time()

    output_dir = os.path.join(OUTPUT_BASE_FOLDER, property_id)
    os.makedirs(output_dir, exist_ok=True)

    # List all images in the specific subfolder
    image_objects = list_s3_images_in_folder(subfolder)

    if not image_objects:
        logger.info(f"[-] No images found in S3 subfolder: {subfolder}")
        return 0.0

    logger.info(f"\n[+] Processing S3 subfolder: {subfolder} | {len(image_objects)} images")
    logger.info(f"[+] S3 subfolder: {S3_BASE_PREFIX}{subfolder}/")
    logger.info(f"[+] Output directory: {output_dir}")

    # Process all images in a single call
    property_cost = 0.0

    try:
        logger.info(f"    [→] Processing all {len(image_objects)} images in single call")

        # Add timeout and error handling
        try:
            logger.debug(f"    [DEBUG] Starting OpenAI API call for all images")
            import signal

            # Set a timeout for the entire processing
            def timeout_handler(signum, frame):
                raise TimeoutError("Processing timed out")

            # Set 10 minute timeout for all images
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)  # 10 minutes

            try:
                result, cost = call_openai_optimized_s3(image_objects, prompt)
                signal.alarm(0)  # Cancel the alarm
                logger.debug(f"    [DEBUG] OpenAI API call completed")
                property_cost += cost
            except TimeoutError as te:
                signal.alarm(0)  # Cancel the alarm
                logger.info(f"    [!] Timeout: {te}")
                return property_cost

            if result:
                logger.debug(f"    [DEBUG] Processing result")

                # Generate or update object files (merge with existing)
                logger.debug(f"    [DEBUG] Generating/updating object files")
                object_files = merge_and_update_object_files_s3(result, image_objects, output_dir, 1)

                # Skip property.json creation as requested

                # Generate individual relationship files
                logger.debug(f"    [DEBUG] Generating individual relationship files")
                property_cid = generate_clean_cid("property", property_id.replace("/", "_").replace(" ", "_"))
                relationship_schema = schemas.get("relationship") if schemas else None
                relationships = generate_individual_relationship_files_s3(object_files, image_objects, property_cid, property_id, relationship_schema, output_dir)

                # Create main relationship file using IPFS schema (merge with existing)
                logger.debug(f"    [DEBUG] Creating/updating main relationship file")
                create_main_relationship_file(relationships, output_dir, property_id)

                logger.debug(f"    [DEBUG] Completed processing category {category}")
            else:
                logger.info(f"    [!] No result for processing")

        except Exception as e:
            logger.error(f"    [!] Error processing: {e}")
            import traceback
            traceback.print_exc()

    except KeyboardInterrupt:
        logger.info(f"    [!] Processing interrupted for {subfolder}")
        return property_cost
    except Exception as e:
        logger.error(f"    [!] Fatal error processing {subfolder}: {e}")
        return property_cost

    elapsed = time.time() - start_time
    logger.info(f"[✓] Done: {subfolder} | 💰 ${property_cost:.4f} | ⏱️ {elapsed:.1f} sec")
    return property_cost


def process_local_category_folder(property_id, category, prompt, schemas=None, batch_size=10, max_workers=12):
    """Process images from local categorized folders with optimized settings for Colab"""
    start_time = time.time()

    try:
        # Local folder path: images/property_id/category/
        local_folder_path = os.path.join("images", property_id, category)

        if not os.path.exists(local_folder_path):
            logger.warning(f"⚠️  Local folder not found: {local_folder_path}")
            return False

        # Get all image files in the local folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        image_files = []

        for file in os.listdir(local_folder_path):
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_path = os.path.join(local_folder_path, file)
                image_files.append(image_path)

        if not image_files:
            logger.warning(f"⚠️  No images found in local folder: {local_folder_path}")
            return False

        logger.info(f"📁 Found {len(image_files)} images in local folder: {local_folder_path}")

        # Load county layout data for this property
        county_layouts = load_county_layout_data(property_id)
        if county_layouts:
            logger.info(f"📋 Loaded {len(county_layouts)} county layout records for property {property_id}")
        else:
            logger.info(f"📋 No county layout data found for property {property_id}")

        # Create property-specific prompt with county data
        property_prompt = load_optimized_json_schema_prompt(category, schemas, county_layouts)

        # Optimize for Colab: Process all images in a single call for speed
        if len(image_files) <= 30:  # For small batches, use single call
            logger.info(f"🚀 Processing all {len(image_files)} images in single call for speed")

            try:
                # Process all images at once
                result, cost = call_openai_optimized(image_files, property_prompt)

                if result:
                    # Generate output files
                    output_dir = os.path.join("output", property_id)
                    os.makedirs(output_dir, exist_ok=True)

                    # Generate individual object files
                    object_files = generate_individual_object_files(result, image_files, output_dir, 1)

                    # Generate individual relationship files
                    property_cid = generate_clean_cid("property", property_id.replace("/", "_").replace(" ", "_"))
                    relationship_schema = schemas.get("relationship") if schemas else None
                    relationships = generate_individual_relationship_files_s3(object_files, image_files, property_cid, property_id, relationship_schema, output_dir)

                    # Create main relationship file
                    create_main_relationship_file(relationships, output_dir, property_id)

                    elapsed = time.time() - start_time
                    logger.info(f"✅ Successfully processed {len(image_files)} images in {elapsed:.1f}s")
                    return True
                else:
                    logger.warning(f"⚠️  No result from single call processing")
                    return False

            except Exception as e:
                logger.error(f"❌ Single call processing failed: {e}")
                # Fall back to batch processing
                logger.info(f"🔄 Falling back to batch processing...")

        # For larger batches, use optimized parallel processing
        logger.info(f"🚀 Processing {len(image_files)} images in {max_workers} parallel batches")

        # Optimize batch size for Colab - respect OpenAI's 10 image limit
        optimal_batch_size = min(batch_size, max(5, len(image_files) // max_workers))
        batches = [image_files[i:i + optimal_batch_size] for i in range(0, len(image_files), optimal_batch_size)]

        logger.info(f"📦 Created {len(batches)} batches of ~{optimal_batch_size} images each")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for batch_num, batch in enumerate(batches, 1):
                future = executor.submit(process_image_batch, batch, property_prompt, batch_num, schemas)
                futures.append(future)

            # Wait for all batches to complete with timeout
            batch_results = []
            completed = 0

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=600)  # 10 minute timeout per batch for Colab
                    if result:
                        batch_results.append(result)
                    completed += 1
                    logger.info(f"✅ Completed batch {completed}/{len(batches)}")
                except Exception as e:
                    logger.error(f"❌ Batch processing failed: {e}")
                    logger.info(f"🔄 Continuing with remaining batches...")

        if batch_results:
            # Merge all batch results
            merged_result = merge_batch_results_intelligently(batch_results)

            # Generate output files
            output_dir = os.path.join("output", property_id)
            os.makedirs(output_dir, exist_ok=True)

            # Generate individual object files
            object_files = generate_individual_object_files(merged_result, image_files, output_dir, 1)

            # Generate individual relationship files
            property_cid = generate_clean_cid("property", property_id.replace("/", "_").replace(" ", "_"))
            relationship_schema = schemas.get("relationship") if schemas else None
            relationships = generate_individual_relationship_files_s3(object_files, image_files, property_cid, property_id, relationship_schema, output_dir)

            # Create main relationship file
            create_main_relationship_file(relationships, output_dir, property_id)

            elapsed = time.time() - start_time
            logger.info(f"✅ Successfully processed {len(image_files)} images in {elapsed:.1f}s")
            return True

        return False

    except Exception as e:
        logger.error(f"❌ Error processing local folder {local_folder_path}: {e}")
        return False

def process_s3_subfolder_multi_threaded(property_id, category, prompt, schemas=None, batch_size=3, max_workers=3):
    """Process a single S3 category folder with true multi-threading and intelligent layout merging."""
    start_time = time.time()

    output_dir = os.path.join(OUTPUT_BASE_FOLDER, property_id)
    os.makedirs(output_dir, exist_ok=True)

    # List all images in the specific category folder
    image_objects = list_s3_images_in_folder(category, property_id)

    if not image_objects:
        logger.info(f"[-] No images found in S3 category: {category}")
        return 0.0

    logger.info(f"\n[+] Processing S3 category: {category} | {len(image_objects)} images")
    logger.info(f"[+] S3 category: {property_id}/{category}/")
    logger.info(f"[+] Output directory: {output_dir}")
    logger.info(f"[+] Two-phase processing: {batch_size} images per batch, {max_workers} workers")

    # Split images into smaller batches for parallel processing
    batches = [image_objects[i:i + batch_size] for i in range(0, len(image_objects), batch_size)]
    logger.info(f"[+] Created {len(batches)} batches of {batch_size} images each")

    # Phase 1: Process batches in parallel for speed
    total_cost = 0.0
    all_batch_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches for processing
        future_to_batch = {
            executor.submit(process_image_batch, batch, prompt, i+1, schemas): i+1
            for i, batch in enumerate(batches)
        }

        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                result, cost = future.result()
                total_cost += cost
                if result:
                    all_batch_results.append(result)
                    logger.info(f"    [✔] Batch {batch_num} completed successfully")
                else:
                    logger.error(f"    [!] Batch {batch_num} failed - no result returned")
            except Exception as e:
                logger.error(f"    [!] Batch {batch_num} failed with error: {e}")
                logger.info(f"    [🔄] Continuing with remaining batches...")

    # Phase 2: Intelligent merging with layout detection
    if all_batch_results:
        logger.debug(f"    [DEBUG] Phase 2: Intelligent merging of {len(all_batch_results)} batch results")
        logger.debug(f"    [DEBUG] All batch results: {[list(result.keys()) for result in all_batch_results]}")
        try:
            merged_result = merge_batch_results_intelligently(all_batch_results, schemas)
            logger.debug(f"    [DEBUG] Merged result keys: {list(merged_result.keys()) if merged_result else 'None'}")
            logger.debug(f"    [DEBUG] Merged result: {json.dumps(merged_result, indent=2)[:1000] if merged_result else 'None'}...")
        except Exception as e:
            logger.error(f"    [!] Error in merge_batch_results_intelligently: {e}")
            import traceback
            traceback.print_exc()
            merged_result = None

        # Generate or update object files (merge with existing)
        logger.debug(f"    [DEBUG] Generating/updating object files")
        logger.debug(f"    [DEBUG] Merged result keys: {list(merged_result.keys())}")
        logger.debug(f"    [DEBUG] Output directory: {output_dir}")
        try:
            object_files = merge_and_update_object_files_s3(merged_result, image_objects, output_dir, 1)
            logger.debug(f"    [DEBUG] Generated object files: {len(object_files) if object_files else 0}")
        except Exception as e:
            logger.error(f"    [!] Error in merge_and_update_object_files_s3: {e}")
            import traceback
            traceback.print_exc()
            object_files = {}

        # Skip property.json creation as requested

        # Generate individual relationship files
        logger.debug(f"    [DEBUG] Generating individual relationship files")
        property_cid = generate_clean_cid("property", property_id.replace("/", "_").replace(" ", "_"))
        relationship_schema = schemas.get("relationship") if schemas else None
        relationships = generate_individual_relationship_files_s3(object_files, image_objects, property_cid, property_id, relationship_schema, output_dir)

        # Create main relationship file using IPFS schema (merge with existing)
        logger.debug(f"    [DEBUG] Creating/updating main relationship file")
        create_main_relationship_file(relationships, output_dir, property_id)

        logger.debug(f"    [DEBUG] Completed processing category {category}")
    else:
        logger.info(f"    [!] No successful results from any batch")

    elapsed = time.time() - start_time
    logger.info(f"[✓] Done: {category} | 💰 ${total_cost:.4f} | ⏱️ {elapsed:.1f} sec")
    return total_cost


def merge_batch_results_intelligently(batch_results, schemas=None):
    """Intelligently merge batch results, detecting and combining duplicate layouts."""
    # Initialize merged structure based on IPFS schemas
    if schemas and isinstance(schemas, dict):
        merged = {}
        for schema_key, schema_data in schemas.items():
            if schema_key != "relationship":
                if schema_data and isinstance(schema_data, dict):
                    # Check if it should be an array or object based on schema
                    if schema_data.get('type') == 'array':
                        merged[schema_key] = []
                    else:
                        merged[schema_key] = {}
                else:
                    # Default to array for most schemas
                    merged[schema_key] = []
    else:
        # Fallback structure
        merged = {
            "layout": [],
            "structure": {},
            "lot": {},
            "utility": [],
            "appliance": []
        }

    # Collect all layouts from all batches
    all_layouts = []
    for batch_result in batch_results:
        if not batch_result or "layout" not in batch_result:
            continue

        layouts = batch_result["layout"]
        if isinstance(layouts, list):
            all_layouts.extend(layouts)
        elif isinstance(layouts, dict):
            all_layouts.append(layouts)
        else:
            logger.debug(f"    [DEBUG] Unexpected layout type: {type(layouts)}")

    # Ensure merged["layout"] is always an array
    if not isinstance(merged.get("layout"), list):
        merged["layout"] = []

    # Group layouts by room type and detect duplicates with better matching
    layout_groups = {}
    for layout in all_layouts:
        if not layout:
            continue

        # Use only fields that exist in the actual IPFS schema
        # We'll group by space_type since that's likely to be in the schema
        space_type = layout.get("space_type", "unknown")

        # Clean the space_type - remove trailing spaces and normalize
        if space_type and isinstance(space_type, str):
            space_type = space_type.strip()
            if not space_type:  # If empty after stripping, use "unknown"
                space_type = "unknown"

        # Create a key for grouping similar layouts
        # For now, use space_type as the primary grouping key
        group_key = space_type

        if group_key not in layout_groups:
            layout_groups[group_key] = []
        layout_groups[group_key].append(layout)

    # Merge layouts within each group
    for group_key, layouts in layout_groups.items():
        if len(layouts) == 1:
            # Single layout, no merging needed
            merged["layout"].append(layouts[0])
        else:
            # Multiple layouts for same room, merge them
            merged_layout = merge_multiple_layouts(layouts)
            merged["layout"].append(merged_layout)

    # Merge other data types
    for batch_result in batch_results:
        if not batch_result:
            continue

        # Merge structure
        if "structure" in batch_result and batch_result["structure"]:
            if not merged["structure"]:
                merged["structure"] = batch_result["structure"]
            else:
                merged["structure"] = merge_structure_data(merged["structure"], batch_result["structure"])

        # Merge lot
        if "lot" in batch_result and batch_result["lot"]:
            if not merged["lot"]:
                merged["lot"] = batch_result["lot"]
            else:
                merged["lot"] = merge_lot_data(merged["lot"], batch_result["lot"])

        # Merge utility
        if "utility" in batch_result:
            # Ensure merged["utility"] is always an array
            if not isinstance(merged.get("utility"), list):
                merged["utility"] = []

            if isinstance(batch_result["utility"], list):
                merged["utility"].extend(batch_result["utility"])
            elif isinstance(batch_result["utility"], dict):
                merged["utility"].append(batch_result["utility"])

        # Merge appliances
        if "appliance" in batch_result:
            # Ensure merged["appliance"] is always an array
            if not isinstance(merged.get("appliance"), list):
                merged["appliance"] = []

            if isinstance(batch_result["appliance"], list):
                merged["appliance"].extend(batch_result["appliance"])
            elif isinstance(batch_result["appliance"], dict):
                merged["appliance"].append(batch_result["appliance"])

    return merged


def merge_multiple_layouts(layouts):
    """Merge multiple layouts that represent the same room."""
    if not layouts:
        return {}

    # Start with the first layout
    merged = layouts[0].copy()

    # Merge additional layouts
    for i, layout in enumerate(layouts[1:], 1):
        merged = merge_layout_data(merged, layout)
    return merged


def process_image_batch(image_batch, prompt, batch_num, schemas=None):
    """Process a single batch of images with OpenAI API."""
    try:
        # Handle both file paths and S3 image objects
        image_objects = []
        for item in image_batch:
            if isinstance(item, dict):
                # S3 image object already in correct format
                image_objects.append(item)
            else:
                # Local file path - convert to image object format
                image_name = os.path.basename(item)
                image_objects.append({
                    'key': item,
                    'name': image_name
                })

        # Process images silently

        result, cost = call_openai_optimized_s3(image_objects, prompt, schemas)
        if result is None:
            logger.info(f"    [!] Batch {batch_num} - OpenAI API returned no result")
        return result, cost
    except Exception as e:
        logger.error(f"    [!] Batch {batch_num} failed with error: {e}")
        return None, 0.0


def merge_batch_results(batch_results):
    """Merge results from multiple batches into a single result."""
    merged = {
        "layout": [],
        "structure": {},
        "lot": {},
        "utility": [],
        "appliance": []
    }

    for batch_result in batch_results:
        if not batch_result:
            continue

        # Merge layouts
        if "layout" in batch_result:
            if isinstance(batch_result["layout"], list):
                merged["layout"].extend(batch_result["layout"])
            elif isinstance(batch_result["layout"], dict):
                merged["layout"].append(batch_result["layout"])

        # Merge structure
        if "structure" in batch_result and batch_result["structure"]:
            if not merged["structure"]:
                merged["structure"] = batch_result["structure"]
            else:
                merged["structure"] = merge_structure_data(merged["structure"], batch_result["structure"])

        # Merge lot
        if "lot" in batch_result and batch_result["lot"]:
            if not merged["lot"]:
                merged["lot"] = batch_result["lot"]
            else:
                merged["lot"] = merge_lot_data(merged["lot"], batch_result["lot"])

        # Merge utility
        if "utility" in batch_result:
            if isinstance(batch_result["utility"], list):
                merged["utility"].extend(batch_result["utility"])
            elif isinstance(batch_result["utility"], dict):
                merged["utility"].append(batch_result["utility"])

        # Merge appliances
        if "appliance" in batch_result:
            if isinstance(batch_result["appliance"], list):
                merged["appliance"].extend(batch_result["appliance"])
            elif isinstance(batch_result["appliance"], dict):
                merged["appliance"].append(batch_result["appliance"])

    return merged


if __name__ == "__main__":
    main()