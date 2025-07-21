# AI Image Analyzer - Google Colab Example
# Copy this into a Google Colab cell

# Install the tool from GitHub
!pip install git+https://github.com/elephant-xyz/photo-meta-data-ai.git --force-reinstall --no-cache-dir

# Set environment variables
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'your-access-key-here'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-secret-key-here'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['S3_BUCKET_NAME'] = 'your-bucket-name-here'
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'

# Run the AI analyzer
!ai-analyzer --all-properties 