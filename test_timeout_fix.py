#!/usr/bin/env python3
"""
Test script to verify timeout fixes work correctly
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, 'src')

from ai_image_analysis_optimized_multi_thread import main

def test_timeout_fixes():
    """Test that the timeout fixes work correctly"""
    print("🧪 Testing timeout fixes...")
    
    # Check if required files exist
    if not os.path.exists('seed.csv'):
        print("❌ seed.csv not found. Please create it first.")
        return False
    
    if not os.path.exists('.env'):
        print("❌ .env file not found. Please create it first.")
        return False
    
    print("✅ Required files found")
    
    # Test with a small batch size and timeout
    print("🚀 Starting test with reduced batch size and increased timeouts...")
    
    try:
        # This will test the timeout fixes
        main()
        print("✅ Test completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_timeout_fixes()
    sys.exit(0 if success else 1) 