#!/bin/bash

# Photo Metadata AI Installation Script
# This script installs the package for remote execution

echo "🚀 Installing Photo Metadata AI package..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.7"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python version $python_version is too old. Please install Python 3.7+"
    exit 1
fi

echo "✅ Python $python_version detected"

# Install the package in development mode
echo "📦 Installing package in development mode..."
python3 -m pip install --upgrade pip
python3 -m pip install -e .

if [ $? -eq 0 ]; then
    echo "✅ Package installed successfully!"
    echo ""
    echo "🎉 Available commands:"
    echo "  photo-categorizer          - AWS Rekognition photo categorization"
    echo "  ai-analyzer                - AI image analysis (optimized)"
    echo "  ai-analyzer-quality        - AI analysis with quality detection"
    echo "  bucket-manager             - S3 bucket management"
    echo "  colab-folder-setup         - Colab folder setup"
    echo "  quality-assessment         - Image quality assessment"
    echo "  upload-to-s3               - Upload files to S3"
    echo "  property-summarizer        - Property data summarization"
    echo "  fix-schema-validation      - Fix schema validation issues"
    echo "  copy-property-files-from-zip - Copy property files from zip"
    echo "  copy-all-files-from-zip    - Copy all files from zip"
    echo "  unzip-count-data           - Unzip and rename county data"
    echo "  copy-all-data-for-submission - Copy all data for submission"
    echo ""
    echo "💡 Example usage:"
    echo "  ai-analyzer --local-folders --parallel-categories --all-properties"
    echo "  fix-schema-validation"
    echo "  copy-all-files-from-zip"
else
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi 