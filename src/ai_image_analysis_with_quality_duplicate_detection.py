#!/usr/bin/env python3
"""
AI Image Analysis with Quality Assessment and Duplicate Detection
Modified version that includes image quality rating and duplicate detection.
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import image as skimage
import cv2
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ai-analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OUTPUT_BASE_FOLDER = "output"
QUALITY_MAPPING_FILE = "quality_mapping.json"


def calculate_image_hash(image_path: str) -> str:
    """Calculate perceptual hash of image for duplicate detection."""
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize for consistent hashing
            img = img.convert('L').resize((8, 8))
            pixels = list(img.getdata())
            
            # Calculate average pixel value
            avg = sum(pixels) / len(pixels)
            
            # Create hash based on pixels above/below average
            hash_str = ''.join(['1' if pixel > avg else '0' for pixel in pixels])
            
            # Convert binary to hex
            return hex(int(hash_str, 2))[2:].zfill(16)
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
        
        # Use SIFT for feature detection
        sift = cv2.SIFT_create()
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


def assess_image_quality(image_path: str) -> Dict[str, Any]:
    """Assess image quality using multiple metrics."""
    try:
        with Image.open(image_path) as img:
            # Basic quality metrics
            width, height = img.size
            aspect_ratio = width / height if height > 0 else 0
            
            # Calculate brightness
            gray_img = img.convert('L')
            brightness = np.mean(np.array(gray_img))
            
            # Calculate contrast
            contrast = np.std(np.array(gray_img))
            
            # Calculate sharpness (using Laplacian variance)
            gray_array = np.array(gray_img)
            laplacian_var = cv2.Laplacian(gray_array, cv2.CV_64F).var()
            
            # Determine quality score (0-10)
            quality_score = 0
            
            # Resolution score (0-3 points)
            if width >= 1920 and height >= 1080:
                quality_score += 3
            elif width >= 1280 and height >= 720:
                quality_score += 2
            elif width >= 640 and height >= 480:
                quality_score += 1
            
            # Brightness score (0-2 points)
            if 50 <= brightness <= 200:
                quality_score += 2
            elif 30 <= brightness <= 220:
                quality_score += 1
            
            # Contrast score (0-2 points)
            if contrast >= 50:
                quality_score += 2
            elif contrast >= 30:
                quality_score += 1
            
            # Sharpness score (0-3 points)
            if laplacian_var >= 100:
                quality_score += 3
            elif laplacian_var >= 50:
                quality_score += 2
            elif laplacian_var >= 20:
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
                "aspect_ratio": round(aspect_ratio, 2),
                "brightness": round(brightness, 1),
                "contrast": round(contrast, 1),
                "sharpness": round(laplacian_var, 1),
                "file_size_mb": round(os.path.getsize(image_path) / (1024 * 1024), 2)
            }
            
    except Exception as e:
        logger.error(f"Error assessing quality for {image_path}: {e}")
        return {
            "quality_score": 0,
            "quality_rating": "error",
            "resolution": "unknown",
            "aspect_ratio": 0,
            "brightness": 0,
            "contrast": 0,
            "sharpness": 0,
            "file_size_mb": 0
        }


def detect_duplicates(image_paths: List[str], similarity_threshold: float = 0.8) -> Dict[str, List[str]]:
    """Detect duplicate images using perceptual hashing and feature similarity."""
    duplicates = {}
    image_hashes = {}
    
    # Calculate hashes for all images
    for img_path in image_paths:
        img_hash = calculate_image_hash(img_path)
        if img_hash:
            if img_hash not in image_hashes:
                image_hashes[img_hash] = []
            image_hashes[img_hash].append(img_path)
    
    # Find exact duplicates (same hash)
    for img_hash, paths in image_hashes.items():
        if len(paths) > 1:
            # All images with same hash are duplicates
            for path in paths:
                duplicates[path] = [p for p in paths if p != path]
    
    # Find similar images (different hash but high similarity)
    for i, img1_path in enumerate(image_paths):
        if img1_path in duplicates:
            continue  # Already marked as duplicate
            
        for j, img2_path in enumerate(image_paths[i+1:], i+1):
            if img2_path in duplicates:
                continue  # Already marked as duplicate
                
            similarity = calculate_image_similarity(img1_path, img2_path)
            if similarity >= similarity_threshold:
                if img1_path not in duplicates:
                    duplicates[img1_path] = []
                if img2_path not in duplicates:
                    duplicates[img2_path] = []
                
                duplicates[img1_path].append(img2_path)
                duplicates[img2_path].append(img1_path)
    
    return duplicates


def create_quality_mapping(image_paths: List[str], output_dir: str) -> Dict[str, Any]:
    """Create quality mapping for all images."""
    quality_mapping = {
        "metadata": {
            "total_images": len(image_paths),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
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
    duplicates = detect_duplicates(image_paths)
    
    # Update quality mapping with duplicate information
    for img_path, duplicate_list in duplicates.items():
        img_name = os.path.basename(img_path)
        if img_name in quality_mapping["images"]:
            quality_mapping["images"][img_name]["is_duplicate"] = True
            quality_mapping["images"][img_name]["duplicate_of"] = [os.path.basename(p) for p in duplicate_list]
            quality_mapping["images"][img_name]["duplicate_count"] = len(duplicate_list)
    
    # Save quality mapping
    mapping_file = os.path.join(output_dir, QUALITY_MAPPING_FILE)
    with open(mapping_file, 'w') as f:
        json.dump(quality_mapping, f, indent=2)
    
    logger.info(f"Quality mapping saved to: {mapping_file}")
    return quality_mapping


def analyze_images_with_quality(image_paths: List[str], output_dir: str, 
                              prompt: str = None, schemas: Dict = None) -> Dict[str, Any]:
    """Analyze images with quality assessment and duplicate detection."""
    
    # Create quality mapping first
    logger.info("Creating quality mapping...")
    quality_mapping = create_quality_mapping(image_paths, output_dir)
    
    # Get quality statistics
    total_images = len(image_paths)
    quality_ratings = {}
    duplicate_count = 0
    
    for img_data in quality_mapping["images"].values():
        rating = img_data["quality_assessment"]["quality_rating"]
        quality_ratings[rating] = quality_ratings.get(rating, 0) + 1
        
        if img_data["is_duplicate"]:
            duplicate_count += 1
    
    # Print quality summary
    print(f"\n{'='*60}")
    print(f"IMAGE QUALITY SUMMARY")
    print(f"{'='*60}")
    print(f"Total Images: {total_images}")
    print(f"Duplicates: {duplicate_count}")
    print(f"\nQuality Distribution:")
    for rating, count in sorted(quality_ratings.items()):
        percentage = (count / total_images) * 100
        print(f"  {rating}: {count} ({percentage:.1f}%)")
    
    # Filter out duplicates for AI analysis (optional)
    non_duplicate_images = []
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        if not quality_mapping["images"][img_name]["is_duplicate"]:
            non_duplicate_images.append(img_path)
    
    print(f"\nNon-duplicate images for AI analysis: {len(non_duplicate_images)}")
    
    # Here you would integrate with your existing AI analysis
    # For now, we'll just return the quality mapping
    return {
        "quality_mapping": quality_mapping,
        "analysis_images": non_duplicate_images,
        "statistics": {
            "total_images": total_images,
            "duplicate_count": duplicate_count,
            "non_duplicate_count": len(non_duplicate_images),
            "quality_distribution": quality_ratings
        }
    }


def main():
    parser = argparse.ArgumentParser(description="AI Image Analysis with Quality Assessment")
    parser.add_argument("--input-dir", required=True, help="Directory containing images to analyze")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--similarity-threshold", type=float, default=0.8, 
                       help="Similarity threshold for duplicate detection (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(args.input_dir).glob(f"*{ext}"))
        image_paths.extend(Path(args.input_dir).glob(f"*{ext.upper()}"))
    
    image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        print(f"❌ No images found in {args.input_dir}")
        return
    
    print(f"📊 Found {len(image_paths)} images to analyze")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze images with quality assessment
    results = analyze_images_with_quality(
        image_paths, 
        args.output_dir,
        similarity_threshold=args.similarity_threshold
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"Quality mapping saved to: {os.path.join(args.output_dir, QUALITY_MAPPING_FILE)}")


if __name__ == "__main__":
    main() 