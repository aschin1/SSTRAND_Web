#!/bin/bash

# SSTRAND Complete Setup Script
echo "🚀 Setting up SSTRAND for real data processing..."
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

print_info "Node.js version: $(node --version)"
print_info "Python version: $(python3 --version)"

# Create directory structure
print_info "Creating directory structure..."
mkdir -p model classifiers models/{membrane,disordered,structured} pdb_files uploads results public/{css,js} views

# Install Node.js dependencies
if [ -f "package.json" ]; then
    print_info "Installing Node.js dependencies..."
    npm install
    print_status "Node.js dependencies installed"
else
    print_warning "package.json not found. Creating basic package.json..."
    cat > package.json << 'EOF'
{
  "name": "sstrand-server",
  "version": "1.0.0",
  "description": "Protein Secondary Structure Prediction Server",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js",
    "setup": "node setup.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "multer": "^1.4.5-lts.1",
    "ejs": "^3.1.9",
    "uuid": "^9.0.0"
  },
  "devDependencies": {
    "nodemon": "^2.0.22"
  }
}
EOF
    npm install
    print_status "Created and installed basic Node.js dependencies"
fi

# Install Python dependencies
if [ -f "requirements.txt" ]; then
    print_info "Installing Python dependencies..."
    python3 -m pip install -r requirements.txt
    print_status "Python dependencies installed"
else
    print_warning "requirements.txt not found. Installing basic dependencies..."
    python3 -m pip install torch transformers biopython numpy pandas requests
    print_status "Basic Python dependencies installed"
fi

# Create EJS templates if setup.js exists
if [ -f "setup.js" ]; then
    print_info "Creating EJS templates..."
    node setup.js
    print_status "EJS templates created"
fi

# Run validation test
if [ -f "test_setup.py" ]; then
    print_info "Running setup validation..."
    if python3 test_setup.py; then
        print_status "Setup validation passed"
    else
        print_warning "Setup validation had some issues"
    fi
fi

# Check for required files
print_info "Checking for required files..."

required_files=(
    "server.js"
    "final_workflow4.py"
    "platform_utils.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    print_error "Missing required files: ${missing_files[*]}"
    print_info "Please ensure all required files are present"
    exit 1
fi

# Check model directories
print_info "Checking model setup..."
model_issues=()

for model_type in membrane disordered structured; do
    model_dir="models/$model_type"
    if [ ! -d "$model_dir" ]; then
        model_issues+=("$model_dir directory missing")
    elif [ ! -f "$model_dir/config.json" ]; then
        model_issues+=("$model_dir/config.json missing")
    elif [ ! -f "$model_dir/pytorch_model.bin" ] && [ ! -f "$model_dir/model.safetensors" ]; then
        model_issues+=("$model_dir model weights missing")
    fi
done

if [ ${#model_issues[@]} -gt 0 ]; then
    print_warning "Model setup issues:"
    for issue in "${model_issues[@]}"; do
        echo "  - $issue"
    done
    print_info "You may need to download or copy your model files"
else
    print_status "Model setup looks good"
fi

# Create a startup script
cat > start_server.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting SSTRAND server..."

# Check if server is already running
if lsof -Pi :5050 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️ Server is already running on port 5050"
    echo "To stop it: lsof -ti:5050 | xargs kill -9"
    exit 1
fi

# Start the server
echo "Starting server on http://localhost:5050"
node server.js
EOF

chmod +x start_server.sh

# Create a test script
cat > test_server.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing SSTRAND server..."

# Test if server is running
if ! lsof -Pi :5050 -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ Server is not running on port 5050"
    echo "Start it with: ./start_server.sh"
    exit 1
fi

echo "✅ Server is running"

# Test health endpoint
echo "Testing health endpoint..."
if curl -s http://localhost:5050/health > /dev/null; then
    echo "✅ Health endpoint accessible"
else
    echo "❌ Health endpoint not accessible"
fi

# Test main page
echo "Testing main page..."
if curl -s http://localhost:5050/ > /dev/null; then
    echo "✅ Main page accessible"
else
    echo "❌ Main page not accessible"
fi

echo "🎉 Basic server tests completed"
echo "Open http://localhost:5050 in your browser to use SSTRAND"
EOF

chmod +x test_server.sh

# Print completion message
echo ""
print_status "Setup completed!"
echo ""
print_info "📋 Next steps:"
echo "1. Ensure your model files are in the models/ directories"
echo "2. Start the server: ./start_server.sh"
echo "3. Test the setup: ./test_server.sh"
echo "4. Open browser: http://localhost:5050"
echo ""
print_info "📁 Files created:"
echo "- start_server.sh (start the server)"
echo "- test_server.sh (test the server)"
echo "- test_insulin.fasta (test sequence)"
echo ""
print_info "🔧 Useful commands:"
echo "- Check server status: lsof -i :5050"
echo "- Stop server: lsof -ti:5050 | xargs kill -9"
echo "- View logs: tail -f nohup.out (if using nohup)"
echo ""

if [ ${#model_issues[@]} -eq 0 ]; then
    print_status "Ready to run! Execute: ./start_server.sh"
else
    print_warning "Fix model issues first, then run: ./start_server.sh"
fi