#!/usr/bin/env python3
"""
Simple test script for Deforum-X-Flux ComfyUI Nodes
Tests basic functionality without requiring full ComfyUI installation
"""

import sys
import os
import torch
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_basic_imports():
    """Test that all nodes can be imported"""
    print("Testing basic imports...")

    try:
        from nodes import (
            # Core Deforum nodes
            DeforumPromptNode,
            DeforumAnimParamsNode,
            DeforumTranslationParamsNode,
            DeforumDiffusionParamsNode,
            DeforumBaseParamsNode,
            DeforumIteratorNode,
            DeforumFluxSampler,
            DeforumConditioningBlendNode,
            DeforumCacheLatentNode,
            DeforumGetCachedLatentNode,
            # Utility functions
            parse_key_frames,
            get_inbetweens,
            sample_from_cv2,
            sample_to_cv2
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_keyframe_parsing():
    """Test keyframe parsing functionality"""
    print("\nTesting keyframe parsing...")
    
    try:
        from nodes import parse_key_frames, get_inbetweens
        
        # Test basic keyframe parsing
        keyframe_string = "0:(1.0), 10:(2.0), 20:(1.5)"
        parsed = parse_key_frames(keyframe_string)
        expected = {0: "1.0", 10: "2.0", 20: "1.5"}
        
        if parsed == expected:
            print("✅ Keyframe parsing works")
        else:
            print(f"❌ Keyframe parsing failed: {parsed} != {expected}")
            return False
            
        # Test interpolation
        interpolated = get_inbetweens(parsed, 21)
        if len(interpolated) == 21:
            print("✅ Keyframe interpolation works")
        else:
            print(f"❌ Interpolation failed: length {len(interpolated)} != 21")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Keyframe test failed: {e}")
        return False

def test_node_creation():
    """Test that nodes can be created"""
    print("\nTesting node creation...")

    try:
        from nodes import DeforumPromptNode, DeforumIteratorNode, DeforumFluxSampler

        # Test prompt node
        prompt_node = DeforumPromptNode()
        input_types = prompt_node.INPUT_TYPES()

        if "required" in input_types and "prompts" in input_types["required"]:
            print("✅ Prompt node created successfully")
        else:
            print("❌ Prompt node structure incorrect")
            return False

        # Test iterator node
        iterator_node = DeforumIteratorNode()
        iterator_input_types = iterator_node.INPUT_TYPES()

        if "required" in iterator_input_types and "deforum_data" in iterator_input_types["required"]:
            print("✅ Iterator node created successfully")
        else:
            print("❌ Iterator node structure incorrect")
            return False

        # Test FLUX sampler
        sampler_node = DeforumFluxSampler()
        sampler_input_types = sampler_node.INPUT_TYPES()

        if "required" in sampler_input_types and "model" in sampler_input_types["required"]:
            print("✅ FLUX sampler node created successfully")
        else:
            print("❌ FLUX sampler node structure incorrect")
            return False

        return True
    except Exception as e:
        print(f"❌ Node creation test failed: {e}")
        return False

def test_tensor_conversion():
    """Test tensor conversion utilities"""
    print("\nTesting tensor conversion...")
    
    try:
        from nodes import sample_from_cv2, sample_to_cv2
        
        # Create test image
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Convert to tensor and back
        tensor = sample_from_cv2(test_image)
        converted_back = sample_to_cv2(tensor)
        
        if tensor.shape[-1] == 3 and converted_back.shape == test_image.shape:
            print("✅ Tensor conversion works")
            return True
        else:
            print(f"❌ Tensor conversion failed: {tensor.shape} -> {converted_back.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Tensor conversion test failed: {e}")
        return False

def test_deforum_data_flow():
    """Test Deforum data flow through parameter nodes"""
    print("\nTesting Deforum data flow...")

    try:
        from nodes import DeforumPromptNode, DeforumAnimParamsNode, DeforumIteratorNode

        # Test prompt node
        prompt_node = DeforumPromptNode()
        result1 = prompt_node.process('0: "test prompt"')

        if not isinstance(result1, tuple) or len(result1) != 1:
            print("❌ Prompt node returned unexpected result")
            return False

        deforum_data = result1[0]
        if not isinstance(deforum_data, dict) or "prompts" not in deforum_data:
            print("❌ Prompt node data structure incorrect")
            return False

        # Test animation params node
        anim_node = DeforumAnimParamsNode()
        result2 = anim_node.process("2D", 30, "wrap", deforum_data)

        if not isinstance(result2, tuple) or len(result2) != 1:
            print("❌ Animation params node returned unexpected result")
            return False

        updated_data = result2[0]
        if "animation_mode" not in updated_data or updated_data["animation_mode"] != "2D":
            print("❌ Animation params not properly added")
            return False

        print("✅ Deforum data flow works")
        return True

    except Exception as e:
        print(f"❌ Deforum data flow test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Deforum-X-Flux Node Tests")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_keyframe_parsing,
        test_node_creation,
        test_tensor_conversion,
        test_deforum_data_flow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Nodes are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
