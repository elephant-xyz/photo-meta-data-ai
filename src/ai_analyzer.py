#!/usr/bin/env python3
"""
AI Image Analysis Command Line Interface
Processes real estate images using OpenAI and IPFS schemas
"""

import os
import sys
import argparse
from ai_image_analysis_optimized_multi_thread import main

def setup_environment():
    """Setup environment variables from .env file if it exists"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Loaded environment variables from .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed, using system environment variables")
    except FileNotFoundError:
        print("⚠️  No .env file found, using system environment variables")

def validate_environment():
    """Validate that required environment variables are set"""
    required_vars = [
        'OPENAI_API_KEY',
        'AWS_ACCESS_KEY_ID', 
        'AWS_SECRET_ACCESS_KEY',
        'AWS_DEFAULT_REGION'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    
    print("✓ All required environment variables are set")
    return True

def main_cli():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='AI Image Analysis for Real Estate Properties',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all properties in S3
  python ai_analyzer.py --all-properties
  
  # Process a specific property
  python ai_analyzer.py --property-id 30434108090030050
  
  # Process with custom batch size and workers
  python ai_analyzer.py --all-properties --batch-size 10 --max-workers 5
  
  # Process with custom output directory
  python ai_analyzer.py --all-properties --output-dir /path/to/output
        """
    )
    
    parser.add_argument('--property-id', type=str, 
                       help='Specific property ID to process')
    parser.add_argument('--all-properties', action='store_true',
                       help='Process all properties in S3')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Batch size for processing (default: 5)')
    parser.add_argument('--max-workers', type=int, default=3,
                       help='Maximum workers for parallel processing (default: 3)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--bucket', type=str,
                       help='S3 bucket name (overrides S3_BUCKET_NAME env var)')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Override bucket name if specified
    if args.bucket:
        os.environ['S3_BUCKET_NAME'] = args.bucket
        print(f"✓ Using S3 bucket: {args.bucket}")
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Set output directory
    os.environ['OUTPUT_BASE_FOLDER'] = args.output_dir
    
    print(f"\n🚀 Starting AI Image Analysis")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔧 Batch size: {args.batch_size}")
    print(f"👥 Max workers: {args.max_workers}")
    
    # Run the main analysis
    try:
        main()
        print("\n✅ Analysis completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_cli() 