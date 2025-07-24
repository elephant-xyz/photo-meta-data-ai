#!/usr/bin/env python3
"""
Image Converter Utility
Provides WebP conversion functionality for optimizing image storage and transmission.
Supports various image formats and maintains quality while reducing file size.
"""

import os
import io
import logging
from PIL import Image
import base64
from typing import Optional, Tuple, Union
import tempfile


class ImageConverter:
    """Handles image format conversions with focus on WebP optimization."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    DEFAULT_WEBP_QUALITY = 85
    MAX_DIMENSION = 4096  # Maximum dimension to prevent memory issues
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the ImageConverter with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
    
    def convert_to_webp(self, 
                       image_input: Union[str, bytes, Image.Image], 
                       quality: int = DEFAULT_WEBP_QUALITY,
                       lossless: bool = False) -> bytes:
        """
        Convert an image to WebP format.
        
        Args:
            image_input: Path to image file, bytes, or PIL Image object
            quality: Quality for lossy compression (1-100)
            lossless: Use lossless compression
            
        Returns:
            WebP image as bytes
        """
        try:
            # Load image based on input type
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                img = Image.open(image_input)
            elif isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                raise ValueError("Invalid image input type")
            
            # Convert RGBA to RGB if necessary (WebP supports RGBA but RGB is more efficient)
            if img.mode in ('RGBA', 'LA') and not lossless:
                # Create a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img, mask=img.split()[1])
                img = background
            elif img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Resize if image is too large
            if max(img.size) > self.MAX_DIMENSION:
                img.thumbnail((self.MAX_DIMENSION, self.MAX_DIMENSION), Image.Resampling.LANCZOS)
                self.logger.info(f"Resized large image to {img.size}")
            
            # Convert to WebP
            output_buffer = io.BytesIO()
            if lossless:
                img.save(output_buffer, format='WEBP', lossless=True)
            else:
                img.save(output_buffer, format='WEBP', quality=quality, method=6)
            
            webp_bytes = output_buffer.getvalue()
            
            # Log compression ratio
            if isinstance(image_input, str):
                original_size = os.path.getsize(image_input)
                webp_size = len(webp_bytes)
                compression_ratio = (1 - webp_size / original_size) * 100
                self.logger.info(f"Converted to WebP: {original_size} -> {webp_size} bytes "
                               f"({compression_ratio:.1f}% reduction)")
            
            return webp_bytes
            
        except Exception as e:
            self.logger.error(f"Error converting image to WebP: {e}")
            raise
    
    def convert_to_webp_base64(self,
                              image_input: Union[str, bytes, Image.Image],
                              quality: int = DEFAULT_WEBP_QUALITY,
                              lossless: bool = False) -> str:
        """
        Convert an image to WebP format and return as base64 string.
        
        Args:
            image_input: Path to image file, bytes, or PIL Image object
            quality: Quality for lossy compression (1-100)
            lossless: Use lossless compression
            
        Returns:
            Base64 encoded WebP image
        """
        webp_bytes = self.convert_to_webp(image_input, quality, lossless)
        return base64.b64encode(webp_bytes).decode('utf-8')
    
    def optimize_for_upload(self,
                           image_path: str,
                           max_size_mb: float = 5.0,
                           target_format: str = 'webp') -> Tuple[bytes, str]:
        """
        Optimize an image for upload by converting and compressing.
        
        Args:
            image_path: Path to the image file
            max_size_mb: Maximum file size in MB
            target_format: Target format ('webp' recommended)
            
        Returns:
            Tuple of (optimized_bytes, format_extension)
        """
        max_size_bytes = max_size_mb * 1024 * 1024
        
        try:
            # First try with high quality
            if target_format.lower() == 'webp':
                optimized = self.convert_to_webp(image_path, quality=90)
                ext = '.webp'
                
                # If still too large, reduce quality iteratively
                quality = 85
                while len(optimized) > max_size_bytes and quality > 20:
                    optimized = self.convert_to_webp(image_path, quality=quality)
                    quality -= 10
                    
                if len(optimized) > max_size_bytes:
                    self.logger.warning(f"Could not compress image below {max_size_mb}MB")
            else:
                # Fallback to original format with compression
                img = Image.open(image_path)
                output_buffer = io.BytesIO()
                
                # Save in original format with optimization
                img_format = img.format or 'JPEG'
                if img_format == 'JPEG':
                    img.save(output_buffer, format=img_format, quality=85, optimize=True)
                else:
                    img.save(output_buffer, format=img_format, optimize=True)
                
                optimized = output_buffer.getvalue()
                ext = os.path.splitext(image_path)[1].lower()
            
            return optimized, ext
            
        except Exception as e:
            self.logger.error(f"Error optimizing image for upload: {e}")
            raise
    
    def batch_convert_to_webp(self,
                             image_paths: list,
                             output_dir: str,
                             quality: int = DEFAULT_WEBP_QUALITY,
                             preserve_names: bool = True) -> list:
        """
        Convert multiple images to WebP format.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save converted images
            quality: Quality for lossy compression
            preserve_names: Keep original filenames (change extension only)
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for image_path in image_paths:
            try:
                if not os.path.exists(image_path):
                    self.logger.warning(f"Skipping non-existent file: {image_path}")
                    continue
                
                # Generate output filename
                if preserve_names:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_filename = f"{base_name}.webp"
                else:
                    output_filename = f"image_{len(output_paths):04d}.webp"
                
                output_path = os.path.join(output_dir, output_filename)
                
                # Convert and save
                webp_bytes = self.convert_to_webp(image_path, quality=quality)
                with open(output_path, 'wb') as f:
                    f.write(webp_bytes)
                
                output_paths.append(output_path)
                self.logger.info(f"Converted: {image_path} -> {output_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to convert {image_path}: {e}")
                continue
        
        return output_paths
    
    @staticmethod
    def is_supported_format(file_path: str) -> bool:
        """Check if file has a supported image format extension."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ImageConverter.SUPPORTED_FORMATS


# Convenience functions for direct use
def convert_to_webp(image_input: Union[str, bytes, Image.Image],
                   quality: int = 85,
                   lossless: bool = False) -> bytes:
    """Convert an image to WebP format."""
    converter = ImageConverter()
    return converter.convert_to_webp(image_input, quality, lossless)


def convert_to_webp_base64(image_input: Union[str, bytes, Image.Image],
                          quality: int = 85,
                          lossless: bool = False) -> str:
    """Convert an image to WebP format and return as base64 string."""
    converter = ImageConverter()
    return converter.convert_to_webp_base64(image_input, quality, lossless)


def optimize_image_for_upload(image_path: str,
                             max_size_mb: float = 5.0) -> Tuple[bytes, str]:
    """Optimize an image for upload."""
    converter = ImageConverter()
    return converter.optimize_for_upload(image_path, max_size_mb)