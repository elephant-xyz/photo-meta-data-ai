import os
import sys
import json
import time
import hashlib
import numpy as np
import cv2
from PIL import Image
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import tempfile
import logging
from dotenv import load_dotenv

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/quality-assessment.log')
        ]
    )
    return logging.getLogger(__name__)

# Load environment variables
def load_environment():
    """Load environment variables from .env file"""
    env_paths = ['.env', '/content/.env', os.path.expanduser('~/.env')]
    
    env_loaded = False
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            env_loaded = True
            break
    
    return env_loaded

load_environment()
logger = setup_logging()

# S3 Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai')
QUALITY_MAPPING_FILE = "quality_mapping.json"

# Initialize S3 client
s3_client = None

def authenticate_aws():
    """Authenticate with AWS S3."""
    global s3_client
    try:
        s3_client = boto3.client('s3')
        # Test the connection
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        logger.info(f"✓ Successfully authenticated with AWS S3 bucket: {S3_BUCKET_NAME}")
        return True
    except NoCredentialsError:
        logger.error("❌ AWS credentials not found. Please configure your AWS credentials.")
        return False
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            logger.error(f"❌ S3 bucket '{S3_BUCKET_NAME}' not found.")
        else:
            logger.error(f"❌ AWS S3 error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error authenticating with AWS: {e}")
        return False

def list_s3_folders():
    """List all folders in S3 bucket."""
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

def list_s3_images_in_folder(folder_name, property_id=None):
    """List all images in a specific S3 folder."""
    try:
        prefix = f"{folder_name}/" if not folder_name.endswith('/') else folder_name
        
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

def download_s3_image_to_temp(s3_key):
    """Download an S3 image to a temporary file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            s3_client.download_file(S3_BUCKET_NAME, s3_key, tmp_file.name)
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error downloading {s3_key}: {e}")
        return None

def calculate_image_hash(image_path: str) -> str:
    """Calculate a hash for the image to detect exact duplicates."""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {image_path}: {e}")
        return ""

def calculate_image_similarity(img1_path: str, img2_path: str) -> float:
    """Calculate similarity between two images using feature matching."""
    try:
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return 0.0
        
        # Use FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # Calculate similarity score
        similarity = len(good_matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0.0
        
        return min(similarity, 1.0)
        
    except Exception as e:
        logger.error(f"Error calculating similarity between {img1_path} and {img2_path}: {e}")
        return 0.0

def assess_image_quality(image_path: str) -> dict:
    """Assess image quality using multiple metrics."""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {
                "quality_score": 0,
                "quality_rating": "error",
                "error": "Could not load image"
            }
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Get file size
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness (average pixel intensity)
        brightness = np.mean(gray)
        
        # Calculate contrast (standard deviation of pixel values)
        contrast = np.std(gray)
        
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate noise level (simplified)
        noise_level = "low" if sharpness > 100 else "medium" if sharpness > 50 else "high"
        
        # Determine quality score (0-10)
        quality_score = 0
        
        # Resolution scoring
        if width >= 1920 and height >= 1080:
            quality_score += 3
        elif width >= 1280 and height >= 720:
            quality_score += 2
        else:
            quality_score += 1
        
        # File size scoring (assuming good compression)
        if file_size_mb >= 1.0:
            quality_score += 2
        else:
            quality_score += 1
        
        # Brightness scoring
        if 50 <= brightness <= 200:
            quality_score += 2
        else:
            quality_score += 1
        
        # Contrast scoring
        if contrast >= 30:
            quality_score += 2
        else:
            quality_score += 1
        
        # Sharpness scoring
        if sharpness >= 100:
            quality_score += 3
        elif sharpness >= 50:
            quality_score += 2
        else:
            quality_score += 1
        
        # Quality rating
        if quality_score >= 8:
            quality_rating = "excellent"
        elif quality_score >= 6:
            quality_rating = "good"
        elif quality_score >= 4:
            quality_rating = "fair"
        elif quality_score >= 2:
            quality_rating = "poor"
        else:
            quality_rating = "very_poor"
        
        return {
            "quality_score": quality_score,
            "quality_rating": quality_rating,
            "resolution": f"{width}x{height}",
            "file_size_mb": round(file_size_mb, 2),
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "sharpness": round(sharpness, 2),
            "noise_level": noise_level
        }
        
    except Exception as e:
        logger.error(f"Error assessing quality for {image_path}: {e}")
        return {
            "quality_score": 0,
            "quality_rating": "error",
            "error": str(e)
        }

def detect_duplicates(image_paths: list, similarity_threshold: float = 0.8) -> dict:
    """Detect duplicate images using hash and similarity comparison."""
    duplicates = {}
    
    # First, detect exact duplicates using hash
    image_hashes = {}
    for img_path in image_paths:
        img_hash = calculate_image_hash(img_path)
        if img_hash in image_hashes:
            # Exact duplicate found
            original_path = image_hashes[img_hash]
            if original_path not in duplicates:
                duplicates[original_path] = []
            duplicates[original_path].append(img_path)
        else:
            image_hashes[img_hash] = img_path
    
    # Then, detect similar images using feature matching
    for i, img1_path in enumerate(image_paths):
        if img1_path in duplicates:
            continue  # Skip if already marked as duplicate
            
        for j, img2_path in enumerate(image_paths[i+1:], i+1):
            if img2_path in duplicates:
                continue  # Skip if already marked as duplicate
                
            similarity = calculate_image_similarity(img1_path, img2_path)
            
            if similarity >= similarity_threshold:
                if img1_path not in duplicates:
                    duplicates[img1_path] = []
                duplicates[img1_path].append(img2_path)
    
    return duplicates

def create_quality_mapping(image_paths: list, similarity_threshold: float = 0.8) -> dict:
    """Create quality mapping for all images (without saving to file)."""
    quality_mapping = {
        "metadata": {
            "total_images": len(image_paths),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "similarity_threshold": similarity_threshold,
            "version": "1.0"
        },
        "images": {}
    }
    
    # Assess quality for each image
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        quality_data = assess_image_quality(img_path)
        
        quality_mapping["images"][img_name] = {
            "file_path": img_path,
            "quality_assessment": quality_data,
            "is_duplicate": False,
            "duplicate_of": [],
            "duplicate_count": 0
        }
    
    # Detect duplicates
    duplicates = detect_duplicates(image_paths, similarity_threshold)
    
    # Update quality mapping with duplicate information
    for img_path, duplicate_list in duplicates.items():
        img_name = os.path.basename(img_path)
        if img_name in quality_mapping["images"]:
            quality_mapping["images"][img_name]["is_duplicate"] = True
            quality_mapping["images"][img_name]["duplicate_of"] = [os.path.basename(p) for p in duplicate_list]
            quality_mapping["images"][img_name]["duplicate_count"] = len(duplicate_list)
    
    return quality_mapping

def process_s3_folder_quality(folder_name, similarity_threshold=0.8):
    """Process quality assessment for a single S3 folder."""
    start_time = time.time()
    
    # Extract property_id from the folder path
    if "/" in folder_name:
        property_id = folder_name.split("/")[0]
    else:
        property_id = folder_name

    # List all images in the S3 folder
    image_objects = list_s3_images_in_folder(folder_name)

    if not image_objects:
        logger.info(f"No images found in S3 folder: {folder_name}")
        return None

    logger.info(f"Processing quality assessment for {len(image_objects)} images in folder: {folder_name}")
    
    # Download images to temp files for quality assessment
    image_paths = []
    for img_obj in image_objects:
        temp_path = download_s3_image_to_temp(img_obj['Key'])
        if temp_path:
            image_paths.append(temp_path)
    
    if not image_paths:
        logger.warning(f"No images could be downloaded from folder: {folder_name}")
        return None
    
    # Create quality mapping
    quality_mapping = create_quality_mapping(image_paths, similarity_threshold)
    
    # Add property information
    quality_mapping["property_id"] = property_id
    quality_mapping["folder_name"] = folder_name
    
    # Print quality summary
    total_images = len(image_paths)
    quality_ratings = {}
    duplicate_count = 0
    
    for img_data in quality_mapping["images"].values():
        rating = img_data["quality_assessment"]["quality_rating"]
        quality_ratings[rating] = quality_ratings.get(rating, 0) + 1
        
        if img_data["is_duplicate"]:
            duplicate_count += 1
    
    logger.info(f"Quality Assessment Summary for {folder_name}:")
    logger.info(f"  Total Images: {total_images}")
    logger.info(f"  Duplicates: {duplicate_count}")
    logger.info(f"  Quality Distribution:")
    for rating, count in sorted(quality_ratings.items()):
        percentage = (count / total_images) * 100
        logger.info(f"    {rating}: {count} ({percentage:.1f}%)")
    
    # Clean up temp files
    for temp_path in image_paths:
        try:
            os.remove(temp_path)
        except:
            pass
    
    processing_time = time.time() - start_time
    logger.info(f"Quality assessment completed in {processing_time:.2f} seconds")
    
    return quality_mapping

def process_all_s3_folders(similarity_threshold=0.8):
    """Process quality assessment for all S3 folders and combine into single JSON."""
    if not authenticate_aws():
        logger.error("Failed to authenticate with AWS. Exiting.")
        return False
    
    folders = list_s3_folders()
    if not folders:
        logger.warning("No folders found in S3 bucket.")
        return False
    
    logger.info(f"Found {len(folders)} folders in S3 bucket")
    
    # Combined quality mapping for all properties
    combined_quality_mapping = {
        "metadata": {
            "total_properties": 0,
            "total_images": 0,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "similarity_threshold": similarity_threshold,
            "version": "1.0"
        },
        "properties": {}
    }
    
    total_processed = 0
    total_images = 0
    
    for folder in folders:
        logger.info(f"\nProcessing folder: {folder}")
        
        quality_mapping = process_s3_folder_quality(folder, similarity_threshold)
        
        if quality_mapping:
            property_id = quality_mapping["property_id"]
            combined_quality_mapping["properties"][property_id] = quality_mapping
            combined_quality_mapping["metadata"]["total_properties"] += 1
            combined_quality_mapping["metadata"]["total_images"] += quality_mapping["metadata"]["total_images"]
            total_processed += 1
            total_images += quality_mapping["metadata"]["total_images"]
            logger.info(f"✓ Successfully processed folder: {folder}")
        else:
            logger.warning(f"⚠ Failed to process folder: {folder}")
    
    # Save combined quality mapping outside output folder
    combined_file = QUALITY_MAPPING_FILE
    with open(combined_file, 'w') as f:
        json.dump(combined_quality_mapping, f, indent=2)
    
    logger.info(f"\n🎉 Quality assessment completed!")
    logger.info(f"📊 Total properties processed: {total_processed}")
    logger.info(f"📊 Total images assessed: {total_images}")
    logger.info(f"📁 Combined results saved to: {combined_file}")
    
    return total_processed > 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quality Assessment and Duplicate Detection for Images')
    parser.add_argument('--folder', type=str, help='Specific S3 folder to process (optional)')
    parser.add_argument('--all-folders', action='store_true', help='Process all folders in S3 bucket')
    parser.add_argument('--similarity-threshold', type=float, default=0.8, 
                       help='Similarity threshold for duplicate detection (0.0-1.0, default: 0.8)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting quality assessment and duplicate detection...")
    logger.info(f"☁️  S3 Bucket: {S3_BUCKET_NAME}")
    logger.info(f"🔍 Similarity Threshold: {args.similarity_threshold}")
    
    if args.folder:
        # Process specific folder
        quality_mapping = process_s3_folder_quality(args.folder, args.similarity_threshold)
        if quality_mapping:
            # Save single property result
            property_id = quality_mapping["property_id"]
            filename = f"quality_mapping_{property_id}.json"
            with open(filename, 'w') as f:
                json.dump(quality_mapping, f, indent=2)
            logger.info(f"✓ Quality assessment completed successfully - saved to {filename}")
        else:
            logger.error("❌ Quality assessment failed")
    elif args.all_folders:
        # Process all folders
        success = process_all_s3_folders(args.similarity_threshold)
        if success:
            logger.info("✓ All folders processed successfully")
        else:
            logger.error("❌ Some folders failed to process")
    else:
        logger.error("Please specify either --folder or --all-folders")
        return False
    
    return True

if __name__ == "__main__":
    main() 