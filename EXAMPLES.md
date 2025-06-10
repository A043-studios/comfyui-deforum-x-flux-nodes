# 🎨 Example Workflows and Tutorials

## 🚀 Quick Start Examples

### Example 1: Basic 2D Animation
**Goal**: Create a simple rotating and zooming animation

**Workflow**:
```
🎬 Animation Setup → 📐 Motion Controller → 🎭 Keyframe Manager → 🎨 Renderer → 📹 Video Output
```

**Settings**:
- **Animation Setup**:
  - Mode: 2D
  - Frames: 60
  - Size: 1024x1024
  - FPS: 12

- **Motion Controller**:
  - Angle: `0:(0), 30:(180), 60:(360)`
  - Zoom: `0:(1.0), 60:(1.2)`
  - Translation X: `0:(0)`
  - Translation Y: `0:(0)`

- **Keyframe Manager**:
  - Positive: `0:(beautiful mountain landscape), 30:(the same landscape at sunset), 60:(landscape under starry night)`
  - Negative: `0:(blurry, low quality, distorted)`

**Expected Result**: 5-second video with smooth rotation and zoom

---

### Example 2: Mathematical Motion Patterns
**Goal**: Create complex motion using mathematical expressions

**Motion Parameters**:
```python
# Smooth oscillating movement
translation_x = "0:(sin(t*0.1)*100)"
translation_y = "0:(cos(t*0.1)*50)"

# Dynamic zoom with breathing effect
zoom = "0:(1.0 + sin(t*0.05)*0.3)"

# Rotating with acceleration
angle = "0:(t*t*0.01)"
```

**Use Case**: Abstract art, music visualizations, hypnotic patterns

---

### Example 3: 3D Camera Movement
**Goal**: Create cinematic 3D camera movements

**Settings**:
- **Animation Setup**: Mode: 3D, Enable depth warping
- **Depth Warping**: Model: MiDaS, FOV: 40

**Motion Parameters**:
```python
# Forward/backward movement
translation_z = "0:(0), 60:(100), 120:(0)"

# Camera rotation
rotation_3d_y = "0:(0), 120:(360)"

# Slight camera shake
rotation_3d_x = "0:(sin(t*0.3)*2)"
```

**Use Case**: Architectural visualization, landscape exploration

---

## 🎬 Advanced Workflows

### Workflow A: Hybrid Video Style Transfer
**Goal**: Transform existing video with AI styles

**Steps**:
1. **Video Input Setup**:
   - Input video: `./input/source_video.mp4`
   - Extract every frame
   - Hybrid motion: Optical Flow

2. **Style Keyframes**:
   ```
   0:(oil painting style)
   25:(watercolor painting)
   50:(digital art style)
   75:(photorealistic)
   100:(impressionist painting)
   ```

3. **Motion Parameters**:
   - Keep original motion from video
   - Add subtle enhancements:
   ```python
   strength_schedule = "0:(0.4), 50:(0.6), 100:(0.4)"
   ```

**Result**: Artistic transformation of existing video

---

### Workflow B: Music-Synchronized Animation
**Goal**: Create animation that follows music rhythm

**Mathematical Expressions** (assuming 120 BPM, 24 FPS):
```python
# Beat synchronization (120 BPM = 2 beats/second = 48 frames/beat)
beat_freq = 0.13  # 48 frames / 360 degrees

# Zoom on beats
zoom = "0:(1.0 + sin(t*beat_freq)*0.2)"

# Rotation following melody
angle = "0:(sin(t*beat_freq*0.5)*45 + t*0.5)"

# Color intensity on beats
strength_schedule = "0:(0.6 + sin(t*beat_freq)*0.2)"
```

**Additional Setup**:
- Add audio track in Video Output node
- Sync animation length to music duration

---

### Workflow C: Architectural Walkthrough
**Goal**: Create smooth architectural visualization

**Camera Path**:
```python
# Smooth forward movement
translation_z = "0:(0), 300:(500)"

# Gentle side-to-side movement
translation_x = "0:(sin(t*0.02)*30)"

# Looking around
rotation_3d_y = "0:(sin(t*0.01)*15 + t*0.1)"

# Slight vertical movement
translation_y = "0:(sin(t*0.015)*10)"
```

**Prompts**:
```
0:(modern architectural interior, clean lines, natural lighting)
100:(the same interior with warm evening lighting)
200:(the interior at night with artificial lighting)
300:(the interior with dramatic shadows)
```

---

## 🧮 Mathematical Expression Library

### Basic Patterns

#### Linear Motion
```python
# Constant speed
translation_x = "0:(t*2)"

# Accelerating
translation_x = "0:(t*t*0.01)"

# Decelerating
translation_x = "0:(100 - (100-t)*(100-t)*0.01)"
```

#### Oscillating Motion
```python
# Simple sine wave
translation_x = "0:(sin(t*0.1)*100)"

# Damped oscillation
translation_x = "0:(sin(t*0.1)*100*exp(-t*0.01))"

# Complex wave
translation_x = "0:(sin(t*0.1)*100 + sin(t*0.3)*30)"
```

#### Circular Motion
```python
# Perfect circle
translation_x = "0:(sin(t*0.1)*100)"
translation_y = "0:(cos(t*0.1)*100)"

# Spiral
translation_x = "0:(sin(t*0.1)*t*0.5)"
translation_y = "0:(cos(t*0.1)*t*0.5)"
```

### Advanced Patterns

#### Easing Functions
```python
# Ease in (slow start)
zoom = "0:(1.0 + (t/100)**2 * 0.5)"

# Ease out (slow end)
zoom = "0:(1.0 + (1-(1-t/100)**2) * 0.5)"

# Ease in-out
zoom = "0:(1.0 + (t/100)**3 * 0.5)" if t < 50 else "0:(1.5 - ((100-t)/100)**3 * 0.5)"
```

#### Noise and Randomness
```python
# Perlin-like noise (approximation)
translation_x = "0:(sin(t*0.1)*100 + sin(t*0.37)*30 + sin(t*0.73)*10)"

# Jitter
angle = "0:(t*2 + sin(t*1.7)*5)"
```

#### Complex Combinations
```python
# Breathing zoom with rotation
zoom = "0:(1.0 + sin(t*0.05)*0.3)"
angle = "0:(t*0.5 + sin(t*0.1)*10)"

# Figure-8 motion
translation_x = "0:(sin(t*0.1)*100)"
translation_y = "0:(sin(t*0.2)*50)"

# Spiral with zoom
translation_x = "0:(sin(t*0.1)*t*0.3)"
translation_y = "0:(cos(t*0.1)*t*0.3)"
zoom = "0:(1.0 + t*0.001)"
```

## 🎯 Use Case Examples

### 1. Product Showcase
**Scenario**: Showcase a product with smooth 360° rotation

```python
# Smooth 360° rotation
rotation_3d_y = "0:(0), 120:(360)"

# Slight zoom in
zoom = "0:(1.0), 120:(1.2)"

# Prompts
prompts = "0:(professional product photography, clean background, studio lighting)"
```

### 2. Nature Documentary Style
**Scenario**: Wildlife or landscape with cinematic movement

```python
# Slow push-in
translation_z = "0:(0), 300:(200)"

# Gentle pan
rotation_3d_y = "0:(0), 300:(30)"

# Prompts with time progression
prompts = """
0:(serene forest clearing in morning light)
100:(the same forest with dappled afternoon sunlight)
200:(the forest in golden hour)
300:(the forest in twilight)
"""
```

### 3. Abstract Art Generation
**Scenario**: Create flowing abstract patterns

```python
# Complex multi-frequency motion
translation_x = "0:(sin(t*0.08)*150 + sin(t*0.23)*50 + sin(t*0.67)*20)"
translation_y = "0:(cos(t*0.08)*150 + cos(t*0.19)*50 + cos(t*0.71)*20)"
angle = "0:(sin(t*0.05)*180 + t*0.3)"
zoom = "0:(1.0 + sin(t*0.03)*0.4 + sin(t*0.11)*0.2)"

# Abstract prompts
prompts = """
0:(flowing liquid colors, abstract art, vibrant)
60:(geometric patterns, kaleidoscope, colorful)
120:(organic forms, fluid dynamics, artistic)
"""
```

### 4. Architectural Visualization
**Scenario**: Interior walkthrough with realistic lighting

```python
# Smooth walkthrough path
translation_z = "0:(0), 240:(400)"
translation_x = "0:(0), 80:(50), 160:(-30), 240:(0)"

# Natural head movement
rotation_3d_y = "0:(0), 60:(15), 120:(-10), 180:(20), 240:(0)"
rotation_3d_x = "0:(sin(t*0.02)*2)"

# Lighting progression
prompts = """
0:(modern interior, natural daylight, architectural photography)
80:(the same interior, warm afternoon light)
160:(the interior with evening ambient lighting)
240:(the interior with dramatic night lighting)
"""
```

## 🔧 Troubleshooting Examples

### Common Expression Errors

#### ❌ Wrong: Missing parentheses
```python
angle = "0:0, 60:360"  # Missing parentheses
```

#### ✅ Correct: Proper format
```python
angle = "0:(0), 60:(360)"  # Correct format
```

#### ❌ Wrong: Invalid function
```python
translation_x = "0:(random()*100)"  # random() not supported
```

#### ✅ Correct: Use supported functions
```python
translation_x = "0:(sin(t*1.7)*100)"  # Use sin for pseudo-random
```

### Performance Optimization Examples

#### For Large Animations (500+ frames):
```python
# Use simpler expressions
angle = "0:(t*0.5)"  # Instead of complex trigonometry

# Reduce parameter complexity
translation_x = "0:(sin(t*0.1)*100)"  # Instead of multiple frequencies
```

#### For Real-time Preview:
- Use preview mode in renderer
- Reduce resolution to 512x512
- Use fewer generation steps (15-20)

## 📚 Learning Path

### Beginner (Week 1)
1. Start with Example 1 (Basic 2D Animation)
2. Experiment with simple linear motion
3. Try different prompt transitions
4. Learn basic mathematical expressions

### Intermediate (Week 2-3)
1. Explore 3D animation with depth warping
2. Create complex motion patterns
3. Experiment with video input workflows
4. Learn advanced mathematical expressions

### Advanced (Week 4+)
1. Create custom mathematical expressions
2. Develop signature animation styles
3. Optimize for performance and quality
4. Contribute to community examples

**Ready to create your masterpiece? Start with Example 1 and let your creativity flow!** 🎨✨
