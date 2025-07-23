#!/usr/bin/env python3
"""
Script to setup Node.js, install Elephant CLI, and validate/upload data folder
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and return the result"""
    print(f"🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description}")
        print(f"Error: {e.stderr}")
        return None

def check_node_version():
    """Check if Node.js 20+ is installed"""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        print(f"📦 Current Node.js version: {version}")
        
        # Extract major version
        major_version = int(version.split('.')[0].replace('v', ''))
        if major_version >= 20:
            print("✅ Node.js 20+ is already installed")
            return True
        else:
            print(f"⚠️ Node.js version {version} is below 20, will upgrade")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js not found, will install")
        return False

def setup_nodejs():
    """Setup Node.js 20+ and npm"""
    print("\n🔧 Setting up Node.js 20+ and npm...")
    
    # Check if we're on macOS
    if sys.platform == "darwin":
        print("🍎 Detected macOS")
        
        # Check if Homebrew is installed
        if run_command("which brew", "Checking Homebrew"):
            print("✅ Homebrew is installed")
            
            # Install Node.js via Homebrew
            if run_command("brew install node@20", "Installing Node.js 20 via Homebrew"):
                # Link the version
                run_command("brew link node@20 --force", "Linking Node.js 20")
            else:
                print("❌ Failed to install Node.js via Homebrew")
                return False
        else:
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
    
    # Check if we're on Linux
    elif sys.platform.startswith("linux"):
        print("🐧 Detected Linux")
        
        # Install Node.js 20
        commands = [
            "curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -",
            "sudo apt-get install -y nodejs",
            "sudo npm install -g n",
            "sudo n 20"
        ]
        
        for cmd in commands:
            if not run_command(cmd, f"Running: {cmd}"):
                return False
    
    else:
        print(f"❌ Unsupported platform: {sys.platform}")
        return False
    
    # Verify installation
    if check_node_version():
        print("✅ Node.js setup completed successfully")
        return True
    else:
        print("❌ Node.js setup failed")
        return False

def install_elephant_cli():
    """Install Elephant CLI globally"""
    print("\n🐘 Installing Elephant CLI...")
    
    if run_command("npm install -g @elephant-xyz/cli", "Installing Elephant CLI"):
        print("✅ Elephant CLI installed successfully")
        return True
    else:
        print("❌ Failed to install Elephant CLI")
        return False

def create_env_file():
    """Create .env file with Elephant and Pinata credentials"""
    print("\n🔐 Creating .env file...")
    
    env_content = """# Elephant and Pinata Configuration
# Update these values with your actual credentials

ELEPHANT_PRIVATE_KEY="your_private_key_here"
PINATA_JWT="your_pinata_jwt_here"
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    print("⚠️  Please update the .env file with your actual credentials:")
    print("   - ELEPHANT_PRIVATE_KEY: Your Elephant private key")
    print("   - PINATA_JWT: Your Pinata JWT token")
    
    return True

def validate_and_upload(data_folder="output"):
    """Run Elephant CLI validation and upload"""
    print(f"\n🔍 Running Elephant CLI validation on {data_folder}...")
    
    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"❌ Data folder '{data_folder}' not found")
        return False
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("❌ .env file not found. Please run setup first.")
        return False
    
    # Run validation with dry-run
    cmd = f"elephant-cli validate-and-upload {data_folder} --dry-run --output-csv test-results.csv"
    
    if run_command(cmd, "Running Elephant CLI validation"):
        print("✅ Validation completed successfully")
        print("📊 Results saved to test-results.csv")
        return True
    else:
        print("❌ Validation failed")
        return False

def main():
    """Main function"""
    print("🚀 Elephant CLI Setup and Validation Script")
    print("=" * 50)
    
    # Step 1: Setup Node.js
    if not setup_nodejs():
        print("❌ Failed to setup Node.js. Exiting.")
        return
    
    # Step 2: Install Elephant CLI
    if not install_elephant_cli():
        print("❌ Failed to install Elephant CLI. Exiting.")
        return
    
    # Step 3: Create .env file
    if not create_env_file():
        print("❌ Failed to create .env file. Exiting.")
        return
    
    # Step 4: Validate and upload
    data_folder = "output"  # Default data folder
    if len(sys.argv) > 1:
        data_folder = sys.argv[1]
    
    if not validate_and_upload(data_folder):
        print("❌ Validation failed. Exiting.")
        return
    
    print("\n🎉 Setup and validation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Update .env file with your actual credentials")
    print("2. Run the validation again without --dry-run when ready")
    print("3. Check test-results.csv for validation results")

if __name__ == "__main__":
    main() 