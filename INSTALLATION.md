# 🚀 Installation Guide

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux
- **Python**: 3.9 or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (12GB+ recommended)
- **RAM**: 16GB+ (32GB recommended for large animations)
- **Storage**: 10GB+ free space

### ComfyUI Installation
Ensure you have ComfyUI installed and working:
```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

## Installation Methods

### Method 1: ComfyUI Manager (Easiest)

1. **Install ComfyUI Manager** (if not already installed):
   - Download from: https://github.com/ltdrdata/ComfyUI-Manager
   - Follow ComfyUI Manager installation instructions

2. **Install Deforum-X-Flux Nodes**:
   - Open ComfyUI
   - Click "Manager" button
   - Search for "Deforum-X-Flux"
   - Click "Install"
   - Restart ComfyUI

### Method 2: Git Clone (Recommended for developers)

```bash
# Navigate to ComfyUI custom nodes directory
cd ComfyUI/custom_nodes/

# Clone the repository
git clone https://github.com/A043-studios/comfyui-deforum-x-flux-nodes.git

# Navigate to the plugin directory
cd comfyui-deforum-x-flux-nodes

# Install dependencies
pip install -r requirements.txt

# Restart ComfyUI
```

### Method 3: Manual Download

1. **Download the latest release**:
   - Go to: https://github.com/A043-studios/comfyui-deforum-x-flux-nodes/releases
   - Download the latest ZIP file

2. **Extract to ComfyUI**:
   ```bash
   # Extract to custom_nodes directory
   unzip comfyui-deforum-x-flux-nodes.zip -d ComfyUI/custom_nodes/
   
   # Navigate to the directory
   cd ComfyUI/custom_nodes/comfyui-deforum-x-flux-nodes/
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Restart ComfyUI**

## Dependency Installation

### Core Dependencies (Required)
```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install opencv-python>=4.5.0
pip install pandas>=1.3.0
pip install numexpr>=2.8.0
pip install numpy>=1.21.0
pip install Pillow>=8.0.0
```

### Optional Dependencies

#### For Video Output (Recommended)
```bash
# Install FFmpeg first
# Windows: Download from https://ffmpeg.org/
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# Then install Python wrapper
pip install ffmpeg-python>=0.2.0
```

#### For Advanced FLUX Features
```bash
pip install transformers>=4.20.0
pip install diffusers>=0.20.0
pip install einops>=0.6.0
```

#### For Advanced 3D Processing
```bash
pip install pytorch3d>=0.7.0
```

## Platform-Specific Instructions

### Windows

1. **Install Python 3.9+** from python.org
2. **Install Git** from git-scm.com
3. **Install Visual Studio Build Tools** (for some dependencies)
4. **Install CUDA** (if using NVIDIA GPU)
5. **Install FFmpeg**:
   - Download from https://ffmpeg.org/download.html
   - Extract to C:\ffmpeg
   - Add C:\ffmpeg\bin to PATH environment variable

### macOS

1. **Install Homebrew** (if not installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install dependencies**:
   ```bash
   brew install python@3.9 git ffmpeg
   ```

3. **Install PyTorch with Metal support**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

### Linux (Ubuntu/Debian)

1. **Update system**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install dependencies**:
   ```bash
   sudo apt install python3.9 python3-pip git ffmpeg
   sudo apt install python3.9-dev build-essential
   ```

3. **Install CUDA** (for NVIDIA GPUs):
   ```bash
   # Follow NVIDIA CUDA installation guide
   # https://developer.nvidia.com/cuda-downloads
   ```

## Verification

### Test Installation
```bash
# Navigate to ComfyUI directory
cd ComfyUI

# Start ComfyUI
python main.py

# Check for Deforum-X-Flux nodes in the node menu
# Look for "Deforum-X-Flux" category
```

### Test Dependencies
```python
# Run this in Python to verify dependencies
import torch
import cv2
import pandas as pd
import numexpr
import numpy as np

print("✅ All core dependencies installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Test FFmpeg
```bash
ffmpeg -version
# Should show FFmpeg version information
```

## Troubleshooting Installation

### Common Issues

#### "No module named 'torch'"
```bash
# Install PyTorch
pip install torch torchvision

# For CUDA support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### "Microsoft Visual C++ 14.0 is required" (Windows)
- Install Visual Studio Build Tools
- Or install Visual Studio Community Edition

#### "FFmpeg not found"
- Ensure FFmpeg is installed and in PATH
- Test with: `ffmpeg -version`

#### "Permission denied" (Linux/macOS)
```bash
# Use sudo for system-wide installation
sudo pip install -r requirements.txt

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

#### "CUDA out of memory"
- Reduce batch size in ComfyUI settings
- Use smaller image resolutions
- Enable model offloading

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv deforum_env

# Activate virtual environment
# Linux/macOS:
source deforum_env/bin/activate
# Windows:
deforum_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

## Performance Optimization

### GPU Setup
- Ensure CUDA drivers are up to date
- Set ComfyUI to use GPU in settings
- Monitor GPU memory usage

### Memory Management
- Close unnecessary applications
- Use appropriate batch sizes
- Enable model offloading for large models

### Storage
- Use SSD for better performance
- Ensure sufficient free space for outputs
- Consider using separate drive for outputs

## Updates

### Updating the Plugin
```bash
cd ComfyUI/custom_nodes/comfyui-deforum-x-flux-nodes/
git pull origin main
pip install -r requirements.txt --upgrade
```

### Checking for Updates
- Watch the GitHub repository for releases
- Use ComfyUI Manager update notifications
- Check changelog for new features

## Support

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Your operating system
   - Python version
   - Error messages
   - Steps to reproduce

**Installation complete! Ready to create amazing animations!** 🎬✨
