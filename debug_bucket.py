#!/usr/bin/env python3
"""
Debug script to check bucket name processing
"""

import os

# Test environment variable reading
print("Environment variable check:")
print(f"S3_BUCKET_NAME env var: '{os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai')}'")
print(f"S3_BUCKET_NAME with rstrip: '{os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai').rstrip('/')}'")

# Test the actual bucket name that would be used
bucket_name = os.getenv('S3_BUCKET_NAME', 'photo-metadata-ai').rstrip('/')
print(f"Final bucket name: '{bucket_name}'")
print(f"Bucket name length: {len(bucket_name)}")
print(f"Bucket name bytes: {bucket_name.encode()}")

# Test if there are any hidden characters
for i, char in enumerate(bucket_name):
    print(f"Char {i}: '{char}' (ord: {ord(char)})") 