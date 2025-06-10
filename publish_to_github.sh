#!/bin/bash

# 🚀 ComfyUI Deforum-X-Flux Nodes - GitHub Publication Script
# This script helps you publish the package to GitHub

set -e  # Exit on any error

echo "🎬 ComfyUI Deforum-X-Flux Nodes - GitHub Publication"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "nodes.py" ] || [ ! -f "README.md" ]; then
    print_error "Please run this script from the comfyui-deforum-x-flux-nodes directory"
    exit 1
fi

print_info "Current directory: $(pwd)"

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

print_status "Git is available"

# Get GitHub username
echo ""
echo "📝 Setup Information"
echo "==================="

if [ -z "$GITHUB_USERNAME" ]; then
    read -p "Enter your GitHub username: " GITHUB_USERNAME
fi

if [ -z "$GITHUB_USERNAME" ]; then
    print_error "GitHub username is required"
    exit 1
fi

print_info "GitHub username: $GITHUB_USERNAME"

# Update URLs in files
echo ""
echo "🔧 Updating URLs..."

# Update README.md
sed -i.bak "s/YOUR_USERNAME/$GITHUB_USERNAME/g" README.md
print_status "Updated README.md"

# Update pyproject.toml
sed -i.bak "s/YOUR_USERNAME/$GITHUB_USERNAME/g" pyproject.toml
print_status "Updated pyproject.toml"

# Update CONTRIBUTING.md
sed -i.bak "s/YOUR_USERNAME/$GITHUB_USERNAME/g" CONTRIBUTING.md
print_status "Updated CONTRIBUTING.md"

# Update INSTALLATION.md
sed -i.bak "s/YOUR_USERNAME/$GITHUB_USERNAME/g" INSTALLATION.md
print_status "Updated INSTALLATION.md"

# Update promotional content
sed -i.bak "s/YOUR_USERNAME/$GITHUB_USERNAME/g" RELEASE_NOTES_v1.0.0.md
sed -i.bak "s/YOUR_USERNAME/$GITHUB_USERNAME/g" PROMOTIONAL_CONTENT.md
print_status "Updated promotional content"

# Remove backup files
rm -f *.bak
print_status "Cleaned up backup files"

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    print_info "Initializing Git repository..."
    git init
    print_status "Git repository initialized"
else
    print_info "Git repository already exists"
fi

# Add all files
echo ""
echo "📦 Preparing files for commit..."
git add .
print_status "All files staged for commit"

# Create initial commit
echo ""
echo "💾 Creating initial commit..."

COMMIT_MESSAGE="🎬 Initial release: ComfyUI Deforum-X-Flux Nodes v1.0.0

✨ Features:
- 8 professional animation nodes
- Mathematical motion control with expressions
- FLUX model integration for high-quality generation
- Professional video output with multiple formats
- Comprehensive documentation and examples
- Complete testing suite with 100% pass rate

🎯 Ready for community use!

🚀 Highlights:
- Mathematical expressions: sin(t*0.1)*100 for smooth motion
- 2D/3D animation modes with depth warping
- Hybrid video composition with optical flow
- Frame interpolation and professional output
- Cross-platform support (Windows/macOS/Linux)
- MIT license for open community development

📚 Documentation:
- Complete installation guides
- Step-by-step tutorials
- Example workflows
- Troubleshooting guides
- Developer contribution guidelines

🧪 Quality Assurance:
- Comprehensive testing suite
- Performance optimization
- Memory management
- Error handling and validation
- Professional code standards

Transform your creative vision into stunning AI-powered animations! 🎬✨"

git commit -m "$COMMIT_MESSAGE"
print_status "Initial commit created"

# Check if remote origin exists
if git remote get-url origin &> /dev/null; then
    print_info "Remote origin already configured"
else
    # Add remote origin
    echo ""
    echo "🔗 Configuring GitHub remote..."
    REPO_URL="https://github.com/$GITHUB_USERNAME/comfyui-deforum-x-flux-nodes.git"
    git remote add origin "$REPO_URL"
    print_status "Remote origin configured: $REPO_URL"
fi

# Set main branch
git branch -M main
print_status "Main branch configured"

# Instructions for GitHub repository creation
echo ""
echo "🌐 GitHub Repository Setup"
echo "=========================="
print_warning "Before pushing, you need to create the repository on GitHub:"
echo ""
echo "1. Go to: https://github.com/new"
echo "2. Repository name: comfyui-deforum-x-flux-nodes"
echo "3. Description: Professional video animation nodes for ComfyUI based on Deforum-X-Flux research"
echo "4. Make it Public (for community access)"
echo "5. DON'T initialize with README, .gitignore, or license (we have them)"
echo "6. Click 'Create repository'"
echo ""

# Ask if repository is created
read -p "Have you created the GitHub repository? (y/n): " REPO_CREATED

if [ "$REPO_CREATED" != "y" ] && [ "$REPO_CREATED" != "Y" ]; then
    print_warning "Please create the GitHub repository first, then run this script again"
    exit 0
fi

# Push to GitHub
echo ""
echo "🚀 Pushing to GitHub..."

if git push -u origin main; then
    print_status "Successfully pushed to GitHub!"
else
    print_error "Failed to push to GitHub. Please check your credentials and repository settings."
    echo ""
    echo "Manual push command:"
    echo "git push -u origin main"
    exit 1
fi

# Success message
echo ""
echo "🎉 SUCCESS! Your ComfyUI Deforum-X-Flux Nodes are now on GitHub!"
echo "=============================================================="
echo ""
echo "📍 Repository URL: https://github.com/$GITHUB_USERNAME/comfyui-deforum-x-flux-nodes"
echo ""
echo "🎯 Next Steps:"
echo "1. 🏷️  Create a release (v1.0.0) on GitHub"
echo "2. 📢 Submit to ComfyUI Manager"
echo "3. 🌟 Share with the community"
echo "4. 📚 Consider creating video tutorials"
echo ""
echo "🔗 Quick Links:"
echo "- Repository: https://github.com/$GITHUB_USERNAME/comfyui-deforum-x-flux-nodes"
echo "- Issues: https://github.com/$GITHUB_USERNAME/comfyui-deforum-x-flux-nodes/issues"
echo "- Releases: https://github.com/$GITHUB_USERNAME/comfyui-deforum-x-flux-nodes/releases"
echo ""
echo "📢 Promotional Content:"
echo "- Check PROMOTIONAL_CONTENT.md for social media posts"
echo "- Use RELEASE_NOTES_v1.0.0.md for release announcement"
echo ""
print_status "Ready to revolutionize AI video animation! 🎬✨"
