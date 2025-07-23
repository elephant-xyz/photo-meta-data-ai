#!/bin/bash

# Elephant CLI Remote Setup Script
# This script can be run on any machine to set up Elephant CLI
# Loads from existing environment variables

set -e  # Exit on any error

echo "🚀 Elephant CLI Remote Setup Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to check environment variables
check_environment() {
    print_status "Checking environment variables..."
    
    local missing_vars=()
    
    # Check for required environment variables
    if [ -z "$ELEPHANT_PRIVATE_KEY" ]; then
        missing_vars+=("ELEPHANT_PRIVATE_KEY")
    fi
    
    if [ -z "$PINATA_JWT" ]; then
        missing_vars+=("PINATA_JWT")
    fi
    
    if [ ${#missing_vars[@]} -eq 0 ]; then
        print_success "All required environment variables are set"
        return 0
    else
        print_warning "Missing environment variables: ${missing_vars[*]}"
        print_status "Please set these variables in your environment or .env file"
        return 1
    fi
}

# Function to install Node.js
install_nodejs() {
    local os=$(detect_os)
    
    print_status "Detected OS: $os"
    
    if command_exists node; then
        local node_version=$(node --version)
        print_success "Node.js already installed: $node_version"
        return 0
    fi
    
    print_status "Installing Node.js..."
    
    case $os in
        "macos")
            if command_exists brew; then
                print_status "Using Homebrew to install Node.js..."
                brew install node
            else
                print_error "Homebrew not found. Please install Homebrew first:"
                print_error "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                return 1
            fi
            ;;
        "linux")
            print_status "Installing Node.js using NodeSource repository..."
            curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
            sudo apt-get install -y nodejs
            ;;
        "windows")
            print_error "Windows detected. Please install Node.js manually from:"
            print_error "  https://nodejs.org/"
            return 1
            ;;
        *)
            print_error "Unsupported OS: $os"
            return 1
            ;;
    esac
    
    if command_exists node; then
        local node_version=$(node --version)
        print_success "Node.js installed successfully: $node_version"
        return 0
    else
        print_error "Failed to install Node.js"
        return 1
    fi
}

# Function to install npm if needed
install_npm() {
    if command_exists npm; then
        local npm_version=$(npm --version)
        print_success "npm already installed: $npm_version"
        return 0
    else
        print_error "npm not found. Please install Node.js which includes npm."
        return 1
    fi
}

# Function to install Elephant CLI
install_elephant_cli() {
    print_status "Installing Elephant CLI..."
    
    if command_exists elephant-cli; then
        print_success "Elephant CLI already installed"
        return 0
    fi
    
    # Install Elephant CLI globally
    if npm install -g @elephant-xyz/cli; then
        print_success "Elephant CLI installed successfully"
        return 0
    else
        print_error "Failed to install Elephant CLI"
        return 1
    fi
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    local errors=0
    
    # Check Node.js
    if command_exists node; then
        local node_version=$(node --version)
        print_success "Node.js: $node_version"
    else
        print_error "Node.js not found"
        ((errors++))
    fi
    
    # Check npm
    if command_exists npm; then
        local npm_version=$(npm --version)
        print_success "npm: $npm_version"
    else
        print_error "npm not found"
        ((errors++))
    fi
    
    # Check Elephant CLI
    if command_exists elephant-cli; then
        print_success "Elephant CLI: Installed"
    else
        print_error "Elephant CLI not found"
        ((errors++))
    fi
    
    if [ $errors -eq 0 ]; then
        print_success "All components installed successfully!"
        return 0
    else
        print_error "Installation verification failed with $errors error(s)"
        return 1
    fi
}

# Function to test Elephant CLI
test_elephant_cli() {
    print_status "Testing Elephant CLI..."
    
    if command_exists elephant-cli; then
        # Test basic functionality
        if elephant-cli --help >/dev/null 2>&1; then
            print_success "Elephant CLI is working correctly"
            return 0
        else
            print_error "Elephant CLI test failed"
            return 1
        fi
    else
        print_error "Elephant CLI not found for testing"
        return 1
    fi
}

# Function to create validation script
create_validation_script() {
    print_status "Creating validation script..."
    
    cat > validate_upload.sh << 'EOF'
#!/bin/bash

# Elephant CLI Validation Script
# This script validates and uploads data to Elephant Network

set -e

echo "🔍 Elephant CLI Validation Script"
echo "================================"

# Check if output directory exists
if [ ! -d "output" ]; then
    echo "❌ Output directory not found!"
    echo "Please run the AI analysis script first to generate output data."
    exit 1
fi

# Check if Elephant CLI is installed
if ! command -v elephant-cli >/dev/null 2>&1; then
    echo "❌ Elephant CLI not found!"
    echo "Please run the setup script first: ./setup_elephant_remote.sh"
    exit 1
fi

echo "✅ Output directory found"
echo "✅ Elephant CLI found"

# List available property directories
echo ""
echo "📁 Available property directories:"
ls -la output/

echo ""
echo "🔍 Running validation on all property directories..."

# Run validation on each property directory
for property_dir in output/*/; do
    if [ -d "$property_dir" ]; then
        property_name=$(basename "$property_dir")
        echo ""
        echo "🏠 Validating property: $property_name"
        
        # Run Elephant CLI validation in dry-run mode
        if elephant-cli validate-and-upload "$property_dir" --dry-run --output-csv "validation-results-$property_name.csv"; then
            echo "✅ Validation completed for $property_name"
            echo "📊 Results saved to validation-results-$property_name.csv"
        else
            echo "❌ Validation failed for $property_name"
        fi
    fi
done

echo ""
echo "🎉 Validation process completed!"
echo "📊 Check the CSV files for detailed results."
EOF

    chmod +x validate_upload.sh
    print_success "Created validation script: validate_upload.sh"
}

# Function to create usage instructions
create_usage_instructions() {
    print_status "Creating usage instructions..."
    
    cat > ELEPHANT_USAGE.md << 'EOF'
# Elephant CLI Usage Guide

## Setup

1. Ensure environment variables are set:
   ```bash
   export ELEPHANT_PRIVATE_KEY="your_private_key_here"
   export PINATA_JWT="your_pinata_jwt_here"
   ```

2. Run the setup script:
   ```bash
   ./setup_elephant_remote.sh
   ```

3. Verify installation:
   ```bash
   node --version
   npm --version
   elephant-cli --help
   ```

## Usage

### Validate and Upload Data

1. **Dry-run validation (recommended first):**
   ```bash
   ./validate_upload.sh
   ```

2. **Validate specific property:**
   ```bash
   elephant-cli validate-and-upload output/PROPERTY_ID --dry-run --output-csv results.csv
   ```

3. **Actual upload (after validation):**
   ```bash
   elephant-cli validate-and-upload output/PROPERTY_ID --output-csv results.csv
   ```

### Command Options

- `--dry-run`: Validate without uploading
- `--output-csv FILE`: Save results to CSV file
- `--help`: Show all options

### File Structure

Your output directory should contain:
```
output/
├── PROPERTY_ID_1/
│   ├── property.json
│   ├── layout.json
│   ├── appliance.json
│   └── relationships.json
├── PROPERTY_ID_2/
│   └── ...
```

### Environment Variables

The script uses these environment variables:
- `ELEPHANT_PRIVATE_KEY`: Your Elephant Network private key
- `PINATA_JWT`: Your Pinata JWT token

### Troubleshooting

1. **Node.js not found:**
   - Run setup script again
   - Install Node.js manually from https://nodejs.org/

2. **Elephant CLI not found:**
   - Run: `npm install -g @elephant-xyz/cli`

3. **Environment variables not set:**
   - Set ELEPHANT_PRIVATE_KEY and PINATA_JWT in your environment
   - Or create a .env file with these variables

4. **Validation errors:**
   - Check JSON file formats
   - Ensure all required fields are present
   - Verify IPFS schema compliance

## Support

For more information, visit:
- Elephant Network: https://elephant.xyz/
- CLI Documentation: https://docs.elephant.xyz/
EOF

    print_success "Created usage instructions: ELEPHANT_USAGE.md"
}

# Main execution
main() {
    echo ""
    print_status "Starting Elephant CLI setup..."
    
    # Check environment variables
    if check_environment; then
        print_success "Environment variables are properly configured"
    else
        print_warning "Some environment variables may be missing"
        print_status "Continuing with setup..."
    fi
    
    # Install Node.js
    if install_nodejs; then
        print_success "Node.js installation completed"
    else
        print_error "Node.js installation failed"
        exit 1
    fi
    
    # Install npm
    if install_npm; then
        print_success "npm installation completed"
    else
        print_error "npm installation failed"
        exit 1
    fi
    
    # Install Elephant CLI
    if install_elephant_cli; then
        print_success "Elephant CLI installation completed"
    else
        print_error "Elephant CLI installation failed"
        exit 1
    fi
    
    # Verify installation
    if verify_installation; then
        print_success "Installation verification passed"
    else
        print_error "Installation verification failed"
        exit 1
    fi
    
    # Test Elephant CLI
    if test_elephant_cli; then
        print_success "Elephant CLI test passed"
    else
        print_error "Elephant CLI test failed"
        exit 1
    fi
    
    # Create additional scripts
    create_validation_script
    create_usage_instructions
    
    echo ""
    print_success "🎉 Elephant CLI setup completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "1. Run the AI analysis script to generate output data"
    echo "2. Use './validate_upload.sh' to validate your data"
    echo "3. Check 'ELEPHANT_USAGE.md' for detailed instructions"
    echo ""
    print_status "Available commands:"
    echo "- elephant-cli --help"
    echo "- ./validate_upload.sh"
    echo "- cat ELEPHANT_USAGE.md"
}

# Run main function
main "$@" 