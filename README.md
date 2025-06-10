# ComfyUI Deforum-X-Flux Nodes

Professional video animation nodes for ComfyUI, based on the original [XmYx/deforum-comfy-nodes](https://github.com/XmYx/deforum-comfy-nodes) with FLUX model integration.

## 🎬 Features

- **Original Deforum Architecture**: Follows the exact node structure and workflow patterns from XmYx/deforum-comfy-nodes
- **FLUX Model Integration**: Enhanced for high-quality generation using state-of-the-art FLUX models
- **Mathematical Motion Control**: Use expressions like `0:(1.0), 30:(1.5*sin(t/10))` for dynamic animations
- **Parameter Chaining**: Modular parameter nodes that chain together like the original
- **Frame-by-Frame Generation**: Iterator-based system with caching for smooth animations
- **Professional Video Output**: Export to MP4, GIF, WebM with FFmpeg integration

## 🚀 Installation

### **Method 1: ComfyUI Manager (Recommended)**
1. Open ComfyUI Manager
2. Search for "Deforum-X-Flux"
3. Click Install
4. Restart ComfyUI

### **Method 2: Manual Installation**
```bash
# Navigate to ComfyUI custom nodes directory
cd ComfyUI/custom_nodes/

# Clone the repository
git clone https://github.com/A043-studios/comfyui-deforum-x-flux-nodes.git

# Install dependencies
cd comfyui-deforum-x-flux-nodes
pip install -r requirements.txt

# Restart ComfyUI
```

### **Method 3: Download and Extract**
1. Download the latest release ZIP
2. Extract to `ComfyUI/custom_nodes/comfyui-deforum-x-flux-nodes/`
3. Install dependencies: `pip install -r requirements.txt`
4. Restart ComfyUI

## 📋 Requirements

### **System Requirements**
- **Python**: 3.9 or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (12GB+ recommended)
- **RAM**: 16GB+ (32GB recommended for large animations)
- **Storage**: 10GB+ free space for models and outputs

### **Software Dependencies**
- **ComfyUI**: Latest version
- **PyTorch**: 2.0.0 or higher
- **FFmpeg**: For video output (install separately)

### **Hardware Recommendations**
| Quality Level | GPU | VRAM | RAM | Performance |
|---------------|-----|------|-----|-------------|
| **Basic** | RTX 3060 | 12GB | 16GB | 2-5 sec/frame |
| **Standard** | RTX 4070 | 16GB | 32GB | 1-3 sec/frame |
| **Professional** | RTX 4090 | 24GB | 64GB | <1 sec/frame |

## 🎮 Quick Start

### **Basic 2D Animation Workflow**

1. **🎬 Add Animation Setup Node**
   - Set animation mode to "2D"
   - Configure 60 frames at 1024x1024
   - Set FPS to 12

2. **📐 Add Motion Controller Node**
   - Connect to Animation Setup
   - Set angle: `0:(0), 30:(180), 60:(360)`
   - Set zoom: `0:(1.0), 60:(1.2)`

3. **🎭 Add Keyframe Manager Node**
   - Connect to Animation Setup
   - Set prompts: `0:(beautiful landscape), 30:(sunset landscape), 60:(night landscape)`

4. **🎨 Add Deforum Renderer Node**
   - Connect all previous nodes
   - Connect your FLUX model, VAE, and CLIP
   - Generate animation frames

5. **📹 Add Video Output Node**
   - Connect rendered images
   - Choose MP4 format
   - Set output path and quality

### **Example Mathematical Expressions**

```python
# Smooth camera movement
translation_x = "0:(sin(t*0.08)*100 + cos(t*0.03)*50)"
translation_y = "0:(cos(t*0.08)*100 + sin(t*0.03)*50)"

# Dynamic zoom with breathing effect
zoom = "0:(1.0 + sin(t*0.05)*0.3 + cos(t*0.02)*0.1)"

# Complex rotation patterns
angle = "0:(sin(t*0.1)*45 + t*0.5)"

# 3D camera movements
rotation_3d_y = "0:(sin(t*0.06)*15 + t*1.5)"
translation_z = "0:(sin(t*0.04)*30)"
```

## 📚 Node Reference

### 🎬 **Deforum Animation Setup**
**Purpose**: Core animation configuration and setup

**Inputs:**
- `animation_mode`: 2D, 3D, Video Input, Interpolation
- `max_frames`: Number of frames (1-10000)
- `width/height`: Output dimensions (64-4096, multiples of 64)
- `fps`: Frame rate (1-60)
- `border_mode`: replicate, wrap
- `use_depth_warping`: Enable 3D depth processing
- `color_coherence`: Frame-to-frame color consistency

**Outputs:**
- `animation_config`: Configuration for other nodes

**Example:**
```
Mode: 2D
Frames: 120
Size: 1024x1024
FPS: 24
```

### 📐 **Deforum Motion Controller**
**Purpose**: Mathematical motion parameter control

**Inputs:**
- `animation_config`: From Animation Setup
- `angle`: Rotation keyframes (degrees)
- `zoom`: Zoom keyframes (scale factor)
- `translation_x/y`: 2D movement keyframes (pixels)
- `translation_z`: 3D depth movement (units)
- `rotation_3d_x/y/z`: 3D rotation keyframes (degrees)
- `noise_schedule`: Noise amount over time (0.0-1.0)
- `strength_schedule`: Generation strength (0.0-1.0)

**Mathematical Variables:**
- `t`: Current frame number
- Standard functions: `sin`, `cos`, `tan`, `log`, `exp`, `sqrt`

**Example Expressions:**
```python
angle = "0:(0), 60:(sin(t*0.1)*180), 120:(360)"
zoom = "0:(1.0), 120:(1.0 + sin(t*0.05)*0.5)"
translation_x = "0:(sin(t*0.08)*200)"
```

### 🎭 **Deforum Keyframe Manager**
**Purpose**: Prompt and parameter scheduling

**Inputs:**
- `animation_config`: Configuration
- `positive_prompts`: Keyframed positive prompts
- `negative_prompts`: Keyframed negative prompts
- `guidance_scale`: CFG scale (1.0-20.0)
- `steps`: Generation steps (1-100)

**Prompt Format:**
```
0:(first prompt), 30:(transition prompt), 60:(final prompt)
```

**Example:**
```
Positive: 0:(serene mountain lake), 40:(lake at golden hour), 80:(lake under starry sky)
Negative: 0:(blurry, low quality, distorted)
```

### 🎨 **Deforum Renderer**
**Purpose**: Main FLUX-based rendering engine

**Inputs:**
- `animation_config`: Configuration
- `motion_params`: Motion data
- `keyframes`: Prompt scheduling
- `flux_model`: FLUX model
- `flux_vae`: VAE for encoding/decoding
- `clip`: CLIP for text encoding
- `init_image`: Optional starting image
- `depth_model`: Optional depth model for 3D

**Outputs:**
- `images`: Generated animation frames
- `animation_data`: Metadata and frame info

**Features:**
- Frame-by-frame generation with motion transformation
- Color coherence across frames
- Preview mode for testing
- Memory-efficient processing

### 🌊 **Deforum Depth Warping**
**Purpose**: 3D depth-aware transformation controller

**Inputs:**
- `animation_config`: Configuration
- `depth_model_name`: midas, adabins, dpt
- `near_plane/far_plane`: 3D camera clipping (200-50000)
- `fov`: Field of view (10-120 degrees)

**Features:**
- MiDaS depth estimation
- 3D perspective transformations
- Depth map visualization
- Camera parameter control

### 🎞️ **Deforum Video Input**
**Purpose**: Video processing and hybrid composition

**Inputs:**
- `animation_config`: Configuration
- `video_path`: Input video file path
- `extract_nth_frame`: Frame extraction rate
- `hybrid_motion`: None, Optical Flow, Perspective, Affine
- `hybrid_flow_method`: DenseRLOF, DIS Medium, Farneback, SF

**Features:**
- Automatic frame extraction
- Optical flow analysis
- Hybrid video composition
- Multiple flow algorithms

### 🔄 **Deforum Interpolation**
**Purpose**: Frame interpolation and smoothing

**Inputs:**
- `animation_data`: From renderer
- `interpolation_method`: linear, cubic, optical_flow
- `interpolation_frames`: Frames to insert (1-10)
- `smooth_transitions`: Enable smoothing
- `preserve_details`: Detail preservation

**Features:**
- Multiple interpolation algorithms
- Configurable frame insertion
- Smooth transition generation
- Detail preservation options

### 📹 **Deforum Video Output**
**Purpose**: Video compilation and export

**Inputs:**
- `images`: Animation frames
- `animation_config`: Configuration
- `output_format`: mp4, gif, webm, mov
- `output_path`: File save location
- `fps`: Output frame rate
- `quality`: low, medium, high, lossless
- `codec`: h264, h265, vp9, prores
- `audio_path`: Optional audio track

**Features:**
- Multiple output formats
- Quality control
- Codec selection
- Audio track support
- FFmpeg integration

## 🎯 Example Workflows

### **Workflow 1: Simple 2D Animation**
```
AnimationSetup(2D, 60 frames) → 
MotionController(rotation + zoom) → 
KeyframeManager(prompt transitions) → 
Renderer(FLUX generation) → 
VideoOutput(MP4)
```

### **Workflow 2: 3D Depth Animation**
```
AnimationSetup(3D, 120 frames) → 
DepthWarping(MiDaS) → 
MotionController(3D movement) → 
KeyframeManager(complex prompts) → 
Renderer(depth-aware) → 
Interpolation(smooth) → 
VideoOutput(high quality)
```

### **Workflow 3: Hybrid Video Composition**
```
AnimationSetup(Video Input) → 
VideoInput(optical flow) → 
MotionController(style transfer) → 
KeyframeManager(artistic styles) → 
Renderer(hybrid composition) → 
VideoOutput(artistic video)
```

## ⚡ Performance Optimization

### **Memory Management**
- Use lower resolutions for testing (512x512)
- Enable model offloading in ComfyUI settings
- Process long animations in segments
- Clear GPU cache between generations

### **Speed Optimization**
- Use preview mode for workflow testing
- Reduce steps for draft generations
- Use efficient mathematical expressions
- Batch process multiple short animations

### **Quality Settings**
- **Draft**: 512x512, 20 steps, 2D mode
- **Preview**: 1024x1024, 30 steps, basic 3D
- **Production**: 1024x1024+, 40+ steps, full 3D + interpolation

## 🐛 Troubleshooting

### **Common Issues**

#### **"FFmpeg not found"**
**Problem**: Video output fails with FFmpeg error
**Solution**: 
```bash
# Windows: Download from https://ffmpeg.org/
# macOS: 
brew install ffmpeg
# Linux:
sudo apt install ffmpeg
```

#### **"Out of memory" errors**
**Problem**: GPU runs out of VRAM during generation
**Solutions**:
- Reduce resolution (1024→512)
- Lower batch size in ComfyUI settings
- Enable model offloading
- Use CPU for some operations
- Close other GPU applications

#### **"Invalid keyframe format"**
**Problem**: Mathematical expressions not parsing correctly
**Solutions**:
- Use format: `0:(value), 10:(value2)`
- Ensure parentheses around values
- Check mathematical syntax
- Use `t` for frame variable
- Avoid unsupported functions

#### **"No frames generated"**
**Problem**: Renderer produces no output
**Solutions**:
- Check max_frames > 0
- Verify FLUX model is loaded
- Ensure prompts are not empty
- Check motion parameters are valid
- Verify all connections are made

#### **Slow generation speed**
**Problem**: Very slow frame generation
**Solutions**:
- Use preview mode for testing
- Reduce image resolution
- Lower generation steps
- Use 2D mode instead of 3D
- Check GPU utilization

### **Performance Issues**

#### **High memory usage**
- Monitor VRAM usage in Task Manager/Activity Monitor
- Use smaller models if available
- Process in smaller batches
- Enable gradient checkpointing

#### **Slow mathematical evaluation**
- Simplify complex expressions
- Avoid nested functions
- Use efficient mathematical operations
- Pre-calculate constants

### **Quality Issues**

#### **Inconsistent frames**
- Enable color coherence
- Use appropriate strength values (0.6-0.8)
- Ensure smooth motion parameters
- Check keyframe transitions

#### **Artifacts in video**
- Increase generation steps
- Use higher quality settings
- Check motion parameter smoothness
- Verify model compatibility

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🙏 Acknowledgments

- **Deforum Team**: Original animation framework
- **XLabs-AI**: FLUX model integration and Deforum-X-Flux research
- **ComfyUI Community**: Node development platform
- **MCP Framework**: Automated development system

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/A043-studios/comfyui-deforum-x-flux-nodes/issues)
- **Discussions**: [GitHub Discussions](https://github.com/A043-studios/comfyui-deforum-x-flux-nodes/discussions)
- **Discord**: [ComfyUI Discord](https://discord.gg/comfyui)

---

**Transform your creative vision into stunning animations with mathematical precision and AI power!** 🎬✨
