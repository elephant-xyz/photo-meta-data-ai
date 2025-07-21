#!/bin/bash

echo "🚀 Installing Photo Metadata AI Categorizer..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

# Install the package in development mode
echo "📦 Installing package..."
pip3 install -e .

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎯 Usage:"
echo "   photo-categorizer"
echo ""
echo "📋 Before running, make sure to set your AWS credentials:"
echo "   export AWS_ACCESS_KEY_ID='your-access-key'"
echo "   export AWS_SECRET_ACCESS_KEY='your-secret-key'"
echo "   export AWS_DEFAULT_REGION='us-east-1'"
echo ""
echo "🔧 Or create a .env file with your credentials" 