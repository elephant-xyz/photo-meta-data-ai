# Photo Metadata AI - AWS Rekognition Photo Categorizer
# Complete Google Colab Code Cell

# Install the tool from GitHub
!pip install git+https://github.com/yourusername/photo-meta-data-ai.git

# Import required libraries
import os
from google.colab import files
import zipfile
import shutil

# Set AWS credentials (replace with your actual credentials)
os.environ['AWS_ACCESS_KEY_ID'] = 'your-access-key-here'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-secret-key-here'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['S3_BUCKET_NAME'] = 'your-bucket-name-here'

# Create images directory structure
!mkdir -p images

# Function to upload and organize images
def setup_images():
    print("Setting up images directory...")
    
    # Upload images (this will open a file picker)
    print("Please upload your images or a zip file containing property folders...")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        if filename.endswith('.zip'):
            # Extract zip file
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('images/')
            print(f"Extracted {filename} to images/")
        else:
            # Move individual files to images directory
            shutil.move(filename, f"images/{filename}")
            print(f"Moved {filename} to images/")
    
    # Show the structure
    !find images -type f | head -20

# Function to run the categorizer
def run_categorizer():
    print("\nRunning Photo Categorizer...")
    print("=" * 50)
    
    # Run the categorizer
    !photo-categorizer

# Main execution
if __name__ == "__main__":
    print("Photo Metadata AI - Starting Setup")
    print("=" * 50)
    
    # Step 1: Setup images
    setup_images()
    
    # Step 2: Run categorizer
    run_categorizer()
    
    print("\nProcess completed!")
    print("Check your S3 bucket for categorized images and results.") 