"""
Standalone test for Deforum-X-Flux core functionality
Tests without ComfyUI dependencies
"""

import torch
import numpy as np
import cv2
import os
import json
import math
import pandas as pd
import re
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import tempfile

# Core utility functions from the nodes
def parse_key_frames(string: str) -> Dict[int, str]:
    """Parse keyframe string format: '0:(value), 10:(value2)'"""
    pattern = r'((?P<frame>[0-9]+):[\s]*\((?P<param>[\S\s]*?)\)([,][\s]?|[\s]?$))'
    frames = {}
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        frames[frame] = param
    return frames

def get_inbetweens(key_frames: Dict[int, str], max_frames: int) -> pd.Series:
    """Interpolate between keyframes"""
    import numexpr
    key_frame_series = pd.Series([np.nan for _ in range(max_frames)])
    
    for i in range(max_frames):
        if i in key_frames:
            value = key_frames[i]
            try:
                # Evaluate mathematical expressions
                t = i  # time variable for expressions
                evaluated = numexpr.evaluate(value) if isinstance(value, str) else float(value)
                key_frame_series[i] = evaluated
            except:
                key_frame_series[i] = float(value) if str(value).replace('.','').isdigit() else 0.0
    
    # Fill and interpolate
    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method='linear', limit_direction='both')
    
    return key_frame_series

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    """Convert CV2 image to tensor"""
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float32)
    return torch.from_numpy(sample)

def sample_to_cv2(sample: torch.Tensor) -> np.ndarray:
    """Convert tensor to CV2 image"""
    if len(sample.shape) == 4:
        sample = sample.squeeze(0)
    sample = sample.permute(1, 2, 0).cpu().numpy()
    sample = ((sample * 0.5) + 0.5).clip(0, 1)
    return (sample * 255).astype(np.uint8)

# Simplified node classes for testing
class TestDeforumAnimationSetup:
    """Test version of DeforumAnimationSetup"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animation_mode": (["2D", "3D", "Video Input", "Interpolation"], {"default": "2D"}),
                "max_frames": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "fps": ("INT", {"default": 12, "min": 1, "max": 60, "step": 1}),
                "border_mode": (["replicate", "wrap"], {"default": "replicate"}),
            },
            "optional": {
                "use_depth_warping": ("BOOLEAN", {"default": True}),
                "diffusion_cadence": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "color_coherence": (["None", "Match Frame 0 RGB", "Match Frame 0 HSV", "Match Frame 0 LAB"], {"default": "Match Frame 0 RGB"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32-1}),
            }
        }
    
    RETURN_TYPES = ("DEFORUM_CONFIG",)
    RETURN_NAMES = ("animation_config",)
    FUNCTION = "setup_animation"
    CATEGORY = "Deforum-X-Flux"
    
    def setup_animation(self, animation_mode, max_frames, width, height, fps, border_mode, 
                       use_depth_warping=True, diffusion_cadence=1, color_coherence="Match Frame 0 RGB", seed=-1):
        
        # Ensure dimensions are multiples of 64 for FLUX
        width = width - (width % 64)
        height = height - (height % 64)
        
        config = {
            "animation_mode": animation_mode,
            "max_frames": max_frames,
            "width": width,
            "height": height,
            "fps": fps,
            "border_mode": border_mode,
            "use_depth_warping": use_depth_warping,
            "diffusion_cadence": diffusion_cadence,
            "color_coherence": color_coherence,
            "seed": seed if seed != -1 else torch.randint(0, 2**32-1, (1,)).item(),
            "device": "cpu",
            "precision": "fp16",
            "batch_size": 1,
        }
        
        return (config,)

class TestDeforumMotionController:
    """Test version of DeforumMotionController"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animation_config": ("DEFORUM_CONFIG",),
                "angle": ("STRING", {"default": "0:(0)", "multiline": True}),
                "zoom": ("STRING", {"default": "0:(1.04)", "multiline": True}),
                "translation_x": ("STRING", {"default": "0:(0)", "multiline": True}),
                "translation_y": ("STRING", {"default": "0:(0)", "multiline": True}),
                "translation_z": ("STRING", {"default": "0:(7.5)", "multiline": True}),
                "rotation_3d_x": ("STRING", {"default": "0:(0)", "multiline": True}),
                "rotation_3d_y": ("STRING", {"default": "0:(0)", "multiline": True}),
                "rotation_3d_z": ("STRING", {"default": "0:(0)", "multiline": True}),
                "noise_schedule": ("STRING", {"default": "0:(0.02)", "multiline": True}),
                "strength_schedule": ("STRING", {"default": "0:(0.65)", "multiline": True}),
                "contrast_schedule": ("STRING", {"default": "0:(1.0)", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("DEFORUM_MOTION",)
    RETURN_NAMES = ("motion_params",)
    FUNCTION = "setup_motion"
    CATEGORY = "Deforum-X-Flux"
    
    def setup_motion(self, animation_config, angle, zoom, translation_x, translation_y,
                    translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z,
                    noise_schedule, strength_schedule, contrast_schedule, **kwargs):
        
        max_frames = animation_config["max_frames"]
        
        # Parse and interpolate all motion parameters
        motion_params = {
            "angle_series": get_inbetweens(parse_key_frames(angle), max_frames),
            "zoom_series": get_inbetweens(parse_key_frames(zoom), max_frames),
            "translation_x_series": get_inbetweens(parse_key_frames(translation_x), max_frames),
            "translation_y_series": get_inbetweens(parse_key_frames(translation_y), max_frames),
            "translation_z_series": get_inbetweens(parse_key_frames(translation_z), max_frames),
            "rotation_3d_x_series": get_inbetweens(parse_key_frames(rotation_3d_x), max_frames),
            "rotation_3d_y_series": get_inbetweens(parse_key_frames(rotation_3d_y), max_frames),
            "rotation_3d_z_series": get_inbetweens(parse_key_frames(rotation_3d_z), max_frames),
            "noise_schedule_series": get_inbetweens(parse_key_frames(noise_schedule), max_frames),
            "strength_schedule_series": get_inbetweens(parse_key_frames(strength_schedule), max_frames),
            "contrast_schedule_series": get_inbetweens(parse_key_frames(contrast_schedule), max_frames),
            "animation_config": animation_config,
        }
        
        return (motion_params,)

# Test functions
def test_utility_functions():
    """Test utility functions"""
    print("🧪 Testing utility functions...")
    
    # Test keyframe parsing
    keyframe_str = "0:(1.0), 10:(2.0), 20:(3.0)"
    result = parse_key_frames(keyframe_str)
    expected = {0: "1.0", 10: "2.0", 20: "3.0"}
    assert result == expected, f"Expected {expected}, got {result}"
    print("  ✅ parse_key_frames: PASSED")
    
    # Test interpolation
    key_frames = {0: "1.0", 10: "2.0"}
    result = get_inbetweens(key_frames, 11)
    assert abs(result[0] - 1.0) < 0.1, f"Expected ~1.0, got {result[0]}"
    assert abs(result[10] - 2.0) < 0.1, f"Expected ~2.0, got {result[10]}"
    assert abs(result[5] - 1.5) < 0.1, f"Expected ~1.5, got {result[5]}"
    print("  ✅ get_inbetweens: PASSED")
    
    # Test tensor conversion
    img_cv2 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    tensor = sample_from_cv2(img_cv2)
    img_back = sample_to_cv2(tensor)
    
    assert tensor.shape == (1, 3, 512, 512), f"Expected (1, 3, 512, 512), got {tensor.shape}"
    assert img_back.shape == (512, 512, 3), f"Expected (512, 512, 3), got {img_back.shape}"
    assert img_back.dtype == np.uint8, f"Expected uint8, got {img_back.dtype}"
    print("  ✅ sample conversion: PASSED")

def test_animation_setup():
    """Test DeforumAnimationSetup node"""
    print("🧪 Testing DeforumAnimationSetup...")
    
    node = TestDeforumAnimationSetup()
    
    # Test input types
    input_types = node.INPUT_TYPES()
    assert "animation_mode" in input_types["required"]
    assert "max_frames" in input_types["required"]
    assert "width" in input_types["required"]
    assert "height" in input_types["required"]
    print("  ✅ INPUT_TYPES structure: PASSED")
    
    # Test setup functionality
    result = node.setup_animation(
        animation_mode="2D",
        max_frames=100,
        width=1024,
        height=1024,
        fps=12,
        border_mode="replicate"
    )
    
    config = result[0]
    assert config["animation_mode"] == "2D"
    assert config["max_frames"] == 100
    assert config["width"] == 1024
    assert config["height"] == 1024
    assert config["fps"] == 12
    assert "seed" in config
    print("  ✅ setup_animation: PASSED")

def test_motion_controller():
    """Test DeforumMotionController node"""
    print("🧪 Testing DeforumMotionController...")
    
    node = TestDeforumMotionController()
    animation_config = {
        "max_frames": 50,
        "animation_mode": "2D",
        "width": 1024,
        "height": 1024
    }
    
    # Test setup
    result = node.setup_motion(
        animation_config=animation_config,
        angle="0:(0), 25:(180), 50:(360)",
        zoom="0:(1.0), 50:(1.2)",
        translation_x="0:(0)",
        translation_y="0:(0)",
        translation_z="0:(0)",
        rotation_3d_x="0:(0)",
        rotation_3d_y="0:(0)",
        rotation_3d_z="0:(0)",
        noise_schedule="0:(0.02)",
        strength_schedule="0:(0.65)",
        contrast_schedule="0:(1.0)"
    )
    
    motion_params = result[0]
    
    # Check that series are created
    assert "angle_series" in motion_params
    assert "zoom_series" in motion_params
    assert "noise_schedule_series" in motion_params
    
    # Check series length
    assert len(motion_params["angle_series"]) == 50
    
    # Check interpolated values
    assert abs(motion_params["angle_series"][0] - 0.0) < 1.0
    assert abs(motion_params["angle_series"][25] - 180.0) < 5.0  # Allow some tolerance
    print("  ✅ setup_motion: PASSED")

def test_mathematical_expressions():
    """Test mathematical expression parsing and evaluation"""
    print("🧪 Testing mathematical expressions...")
    
    # Test simple expressions
    test_cases = [
        ("0:(1.0)", {0: "1.0"}),
        ("0:(sin(t*0.1))", {0: "sin(t*0.1)"}),
        ("0:(1.0), 10:(2.0)", {0: "1.0", 10: "2.0"}),
        ("0:(1.0 + sin(t*0.1)*0.5)", {0: "1.0 + sin(t*0.1)*0.5"})
    ]
    
    for expr, expected in test_cases:
        result = parse_key_frames(expr)
        assert result == expected, f"Expression '{expr}' failed: expected {expected}, got {result}"
    
    # Test mathematical evaluation
    key_frames = {0: "sin(t*0.1)", 10: "cos(t*0.1)"}
    result = get_inbetweens(key_frames, 11)
    assert len(result) == 11, f"Expected 11 frames, got {len(result)}"
    
    print("  ✅ Mathematical expression parsing: PASSED")

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Deforum-X-Flux Standalone Tests")
    print("=" * 50)
    
    try:
        test_utility_functions()
        test_animation_setup()
        test_motion_controller()
        test_mathematical_expressions()
        
        print("=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Core functionality is working correctly")
        return True
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
