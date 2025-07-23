#!/usr/bin/env python3
"""
Ultra-fast AI Image Analysis for Google Colab
Optimized for maximum speed on Colab's limited resources
"""

import os
import json
import time
import base64
import io
import uuid
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from PIL import Image
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ultra-fast settings for Colab
MAX_IMAGE_SIZE = (512, 512)  # Much smaller for speed
JPEG_QUALITY = 60  # Lower quality for faster uploads
BATCH_SIZE = 5  # Smaller batches for faster processing
MAX_WORKERS = 2  # Fewer workers for Colab's CPU
TIMEOUT = 300  # 5 minute timeout
SINGLE_CALL_THRESHOLD = 15  # Use single call for smaller folders

# IPFS Schema CIDs (simplified for speed)
IPFS_SCHEMA_CIDS = {
    "layout": "bafkreih6j2niabid6xkhzdxhhqbq6g5q7a26yj4vazmtjzmctvp2r7bv4e",
    "structure": "bafkreih6j2niabid6xkhzdxhhqbq6g5q7a26yj4vazmtjzmctvp2r7bv4e",
    "lot": "bafkreih6j2niabid6xkhzdxhhqbq6g5q7a26yj4vazmtjzmctvp2r7bv4e",
    "utility": "bafkreih6j2niabid6xkhzdxhhqbq6g5q7a26yj4vazmtjzmctvp2r7bv4e",
    "appliance": "bafkreih6j2niabid6xkhzdxhhqbq6g5q7a26yj4vazmtjzmctvp2r7bv4e"
}

RELATIONSHIP_SCHEMA_CID = "bafkreih6j2niabid6xkhzdxhhqbq6g5q7a26yj4vazmtjzmctvp2r7bv4e"

def setup_logging():
    """Setup logging for Colab"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('colab_fast_analysis.log')
        ]
    )

def load_environment():
    """Load environment variables"""
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set these in your .env file or Colab environment")
        return False
    
    return True

def optimize_image_fast(image_path):
    """Ultra-fast image optimization for Colab"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Aggressive size reduction
            img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
            buffer.seek(0)

            b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return b64_data if len(b64_data) > 0 else None
    except Exception as e:
        logger.error(f"Failed to optimize image {image_path}: {e}")
        return None

def load_schemas_fast():
    """Load schemas with minimal validation for speed"""
    schemas = {}
    
    # Use simplified schemas for speed
    for schema_name in IPFS_SCHEMA_CIDS.keys():
        schemas[schema_name] = {"type": "object", "properties": {}}
    
    schemas["relationship"] = {"type": "object", "properties": {}}
    
    return schemas

def create_fast_prompt():
    """Create a simplified prompt for faster processing"""
    return """Analyze these real estate images and extract key property information. Focus on:

LAYOUT: Room types (kitchen, bedroom, bathroom, living room, etc.)
STRUCTURE: Building type, construction, condition
LOT: Lot size, landscaping, outdoor features
UTILITY: Heating, cooling, electrical, plumbing
APPLIANCE: Appliances present and their types

Return JSON with these sections. Be concise and accurate."""

def call_openai_fast(image_paths, prompt):
    """Ultra-fast OpenAI API call optimized for Colab"""
    try:
        # Prepare images
        image_objects = []
        for image_path in image_paths:
            b64_data = optimize_image_fast(image_path)
            if b64_data:
                image_objects.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_data}"
                    }
                })
        
        if not image_objects:
            logger.error("No valid images to process")
            return None, 0.0
        
        # Make API call
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze these property images:"},
                        *image_objects
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        
        # Parse JSON
        try:
            parsed_result = json.loads(result)
            return parsed_result, 0.0  # Cost tracking simplified
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response")
            return None, 0.0
            
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return None, 0.0

def process_images_fast(image_files, output_dir, property_id):
    """Ultra-fast image processing for Colab"""
    start_time = time.time()
    
    try:
        # Use single call for all images if small enough
        if len(image_files) <= SINGLE_CALL_THRESHOLD:
            logger.info(f"🚀 Processing {len(image_files)} images in single call")
            
            prompt = create_fast_prompt()
            result, cost = call_openai_fast(image_files, prompt)
            
            if result:
                # Save result
                output_file = os.path.join(output_dir, f"fast_analysis_{property_id}.json")
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                elapsed = time.time() - start_time
                logger.info(f"✅ Fast processing completed in {elapsed:.1f}s")
                return True
            else:
                logger.error("❌ Fast processing failed")
                return False
        
        # For larger batches, use minimal parallel processing
        logger.info(f"🚀 Processing {len(image_files)} images in batches")
        
        # Create small batches
        batches = [image_files[i:i + BATCH_SIZE] for i in range(0, len(image_files), BATCH_SIZE)]
        
        all_results = []
        prompt = create_fast_prompt()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            for batch in batches:
                future = executor.submit(call_openai_fast, batch, prompt)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=TIMEOUT)
                    if result and result[0]:
                        all_results.append(result[0])
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
        
        if all_results:
            # Merge results (simplified)
            merged_result = {}
            for result in all_results:
                for key, value in result.items():
                    if key not in merged_result:
                        merged_result[key] = value
                    elif isinstance(value, list) and isinstance(merged_result[key], list):
                        merged_result[key].extend(value)
            
            # Save merged result
            output_file = os.path.join(output_dir, f"fast_analysis_{property_id}.json")
            with open(output_file, 'w') as f:
                json.dump(merged_result, f, indent=2)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Fast processing completed in {elapsed:.1f}s")
            return True
        else:
            logger.error("❌ No results from batch processing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Fast processing failed: {e}")
        return False

def process_local_folder_fast(property_id, category):
    """Process a single local folder with ultra-fast settings"""
    try:
        # Local folder path
        local_folder_path = os.path.join("images", property_id, category)
        
        if not os.path.exists(local_folder_path):
            logger.warning(f"⚠️  Folder not found: {local_folder_path}")
            return False
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = []
        
        for file in os.listdir(local_folder_path):
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_path = os.path.join(local_folder_path, file)
                image_files.append(image_path)
        
        if not image_files:
            logger.warning(f"⚠️  No images found in: {local_folder_path}")
            return False
        
        logger.info(f"📁 Processing {len(image_files)} images from {category}")
        
        # Create output directory
        output_dir = os.path.join("output", property_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process images
        success = process_images_fast(image_files, output_dir, property_id)
        
        if success:
            logger.info(f"✅ Successfully processed {category}")
            return True
        else:
            logger.error(f"❌ Failed to process {category}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error processing {category}: {e}")
        return False

def process_property_fast(property_id):
    """Process a single property with ultra-fast settings"""
    logger.info(f"🏠 Processing property: {property_id}")
    
    # Check if property folder exists
    property_path = os.path.join("images", property_id)
    if not os.path.exists(property_path):
        logger.error(f"❌ Property folder not found: {property_path}")
        return False
    
    # Get categories
    categories = []
    for item in os.listdir(property_path):
        item_path = os.path.join(property_path, item)
        if os.path.isdir(item_path):
            categories.append(item)
    
    if not categories:
        logger.error(f"❌ No category folders found for {property_id}")
        return False
    
    logger.info(f"📁 Found categories: {', '.join(categories)}")
    
    # Process each category
    success_count = 0
    for category in categories:
        if process_local_folder_fast(property_id, category):
            success_count += 1
    
    logger.info(f"✅ Completed {property_id}: {success_count}/{len(categories)} categories processed")
    return success_count > 0

def main():
    """Main function for ultra-fast Colab processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra-fast AI Image Analysis for Colab')
    parser.add_argument('--property-id', required=True, help='Property ID to process')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    
    if not load_environment():
        logger.error("❌ Environment setup failed")
        return
    
    logger.info("🚀 Starting ultra-fast Colab analysis...")
    logger.info(f"⚡ Optimized for speed: {MAX_IMAGE_SIZE[0]}x{MAX_IMAGE_SIZE[1]} images, {JPEG_QUALITY}% quality")
    logger.info(f"📦 Batch size: {BATCH_SIZE}, Workers: {MAX_WORKERS}")
    
    # Process property
    start_time = time.time()
    success = process_property_fast(args.property_id)
    total_time = time.time() - start_time
    
    if success:
        logger.info(f"🎉 Analysis completed successfully in {total_time:.1f}s")
    else:
        logger.error(f"❌ Analysis failed after {total_time:.1f}s")

if __name__ == "__main__":
    main() 