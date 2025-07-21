# Google Colab Command: Setup Images Folder Structure
# Copy this into a separate cell in your Google Colab

import os
from google.colab import files
import zipfile
import shutil

# Set environment variables for folder structure
os.environ['IMAGES_DIR'] = 'images'  # Change this to your preferred folder name
os.environ['S3_BUCKET_NAME'] = 'your-bucket-name-here'  # Set your S3 bucket name

# Create the images directory
images_dir = os.getenv('IMAGES_DIR', 'images')
!mkdir -p {images_dir}

print(f"Created images directory: {images_dir}")
print("=" * 50)

# Function to upload and organize images
def setup_property_folders():
    print("Setting up property folders...")
    
    # Option 1: Upload a zip file with property folders
    print("Option 1: Upload a ZIP file containing property folders")
    print("Option 2: Upload individual images (will be placed in 'default' folder)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        # Upload and extract zip file
        print("Please upload a ZIP file containing property folders...")
        uploaded = files.upload()
        
        for filename in uploaded.keys():
            if filename.endswith('.zip'):
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall(images_dir)
                print(f"Extracted {filename} to {images_dir}/")
            else:
                print(f"Warning: {filename} is not a ZIP file")
    
    elif choice == "2":
        # Upload individual images to a default folder
        print("Please upload individual images...")
        uploaded = files.upload()
        
        # Create default property folder
        default_folder = f"{images_dir}/property-default"
        !mkdir -p {default_folder}
        
        for filename in uploaded.keys():
            if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']):
                shutil.move(filename, f"{default_folder}/{filename}")
                print(f"Moved {filename} to {default_folder}/")
            else:
                print(f"Warning: {filename} is not an image file")
    
    else:
        print("Invalid choice")

# Function to show current structure
def show_structure():
    print("\nCurrent folder structure:")
    !find {images_dir} -type f | head -20
    print(f"\nTotal files in {images_dir}:")
    !find {images_dir} -type f | wc -l

# Run the setup
if __name__ == "__main__":
    print("Photo Metadata AI - Folder Setup")
    print("=" * 50)
    
    # Setup folders
    setup_property_folders()
    
    # Show structure
    show_structure()
    
    print("\nFolder setup completed!")
    print(f"Your images are now in: {images_dir}/")
    print("Ready to run photo-categorizer!") 