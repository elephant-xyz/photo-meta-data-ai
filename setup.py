from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="photo-metadata-ai",
    version="1.0.0",
    author="Photo Metadata AI",
    description="AWS Rekognition photo categorization tool for real estate images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "boto3>=1.26.0",
        "botocore>=1.29.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "Pillow>=10.0.0",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "photo-categorizer=src.rekognition:main",
            "ai-analyzer=src.ai_image_analysis_optimized_multi_thread:main",
            "bucket-manager=src.bucket_manager:main",
            "colab-folder-setup=src.colab_folder_setup:main",
            "quality-assessment=src.quality_assessment:main",
            "upload-to-s3=src.uploadtoS3:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 