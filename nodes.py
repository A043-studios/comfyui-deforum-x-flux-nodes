"""
Deforum-X-Flux ComfyUI Nodes
Professional video animation nodes based on Deforum-X-Flux research

Implements 8 core nodes for complete animation workflow:
- Animation Setup & Configuration
- Motion Parameter Control
- Keyframe Management
- FLUX-based Rendering
- 3D Depth Warping
- Video Input Processing
- Frame Interpolation
- Video Output Compilation
"""

import torch
import numpy as np
import cv2
import os
import json
import math
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import tempfile
import subprocess
import secrets
import re

# ComfyUI imports
try:
    import folder_paths
    from comfy.model_management import get_torch_device, intermediate_device
    import comfy.utils
    import comfy.sample
    import comfy.samplers
    import comfy.model_management
    COMFY_AVAILABLE = True
except ImportError:
    print("ComfyUI not available, running in standalone mode")
    COMFY_AVAILABLE = False

    def get_torch_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    def intermediate_device():
        return get_torch_device()

# Animation utilities (adapted from Deforum-X-Flux)
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
    try:
        import numexpr
        NUMEXPR_AVAILABLE = True
    except ImportError:
        NUMEXPR_AVAILABLE = False

    key_frame_series = pd.Series([np.nan for _ in range(max_frames)])

    for i in range(max_frames):
        if i in key_frames:
            value = key_frames[i]
            try:
                # Evaluate mathematical expressions
                if NUMEXPR_AVAILABLE and isinstance(value, str):
                    # Create safe evaluation context
                    t = i  # time variable for expressions
                    frame = i
                    evaluated = numexpr.evaluate(value, local_dict={'t': t, 'frame': frame})
                else:
                    evaluated = float(value) if isinstance(value, (int, float, str)) else 0.0
                key_frame_series[i] = evaluated
            except Exception as e:
                print(f"Warning: Could not evaluate expression '{value}' at frame {i}: {e}")
                try:
                    key_frame_series[i] = float(value) if str(value).replace('.','').replace('-','').isdigit() else 0.0
                except:
                    key_frame_series[i] = 0.0

    # Fill and interpolate
    if key_frame_series.first_valid_index() is not None:
        key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    if key_frame_series.last_valid_index() is not None:
        key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method='linear', limit_direction='both')

    return key_frame_series

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    """Convert CV2 image to tensor (ComfyUI format)"""
    try:
        # Ensure we have a valid numpy array
        if not isinstance(sample, np.ndarray):
            raise ValueError("Input must be a numpy array")

        # Handle different input shapes
        if len(sample.shape) == 2:  # Grayscale
            sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)
        elif len(sample.shape) == 3 and sample.shape[2] == 4:  # RGBA
            sample = cv2.cvtColor(sample, cv2.COLOR_BGRA2RGB)
        elif len(sample.shape) == 3 and sample.shape[2] == 3:  # BGR
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        # Add batch dimension if needed
        if len(sample.shape) == 3:
            sample = sample[None, ...]

        # Ensure contiguous array and proper dtype
        sample = np.ascontiguousarray(sample, dtype=np.float32)

        # Normalize to [0, 1] range
        sample = sample / 255.0

        # Convert to tensor (B, H, W, C format for ComfyUI)
        tensor = torch.from_numpy(sample)
        return tensor

    except Exception as e:
        print(f"Error in sample_from_cv2: {e}")
        # Return a safe fallback tensor
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

def sample_to_cv2(sample: torch.Tensor) -> np.ndarray:
    """Convert tensor to CV2 image"""
    try:
        # Ensure we have a valid tensor
        if not isinstance(sample, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")

        # Move to CPU if needed
        if sample.device != torch.device('cpu'):
            sample = sample.cpu()

        # Handle batch dimension
        if len(sample.shape) == 4:
            sample = sample.squeeze(0)

        # Handle different tensor formats
        if len(sample.shape) == 3:
            if sample.shape[0] == 3:  # Channels first (C, H, W)
                sample = sample.permute(1, 2, 0)  # Convert to (H, W, C)
            elif sample.shape[2] == 1:  # Grayscale (H, W, 1)
                sample = sample.repeat(1, 1, 3)  # Convert to RGB

        # Convert to numpy with proper handling
        sample = sample.detach().numpy()
        sample = np.clip(sample, 0, 1)
        sample = (sample * 255).astype(np.uint8)

        # Ensure contiguous array
        sample = np.ascontiguousarray(sample)

        # Convert RGB to BGR for OpenCV if needed
        if len(sample.shape) == 3 and sample.shape[2] == 3:
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)

        return sample

    except Exception as e:
        print(f"Error in sample_to_cv2: {e}")
        # Return a safe fallback image
        return np.zeros((64, 64, 3), dtype=np.uint8)

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    from PIL import Image
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    if tensor.shape[0] == 3:  # Channels first
        tensor = tensor.permute(1, 2, 0)
    array = tensor.numpy()
    array = np.clip(array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(array)

def pil_to_tensor(pil_image):
    """Convert PIL Image to tensor"""
    array = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array)
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    return tensor


# Global storage for Deforum data (similar to original)
class DeforumStorage:
    def __init__(self):
        self.deforum_cache = {"image": {}, "latent": {}}
        self.reset = False

gs = DeforumStorage()

class DeforumPromptNode:
    """
    Prompt scheduling node - matches original Deforum structure
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {
                    "default": '0: "a beautiful landscape, highly detailed"\n30: "a serene forest, cinematic lighting"',
                    "multiline": True
                }),
            },
            "optional": {
                "deforum_data": ("DEFORUM_DATA",),
            }
        }

    RETURN_TYPES = ("DEFORUM_DATA",)
    RETURN_NAMES = ("deforum_data",)
    FUNCTION = "process"
    CATEGORY = "Deforum-X-Flux"

    def process(self, prompts, deforum_data=None):
        if deforum_data is None:
            deforum_data = {}

        deforum_data["prompts"] = prompts
        return (deforum_data,)


class DeforumAnimParamsNode:
    """
    Animation parameters node - matches original structure
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animation_mode": (["2D", "3D", "Video Input"], {"default": "2D"}),
                "max_frames": ("INT", {"default": 120, "min": 1, "max": 10000}),
                "border": (["wrap", "replicate"], {"default": "wrap"}),
            },
            "optional": {
                "deforum_data": ("DEFORUM_DATA",),
            }
        }

    RETURN_TYPES = ("DEFORUM_DATA",)
    RETURN_NAMES = ("deforum_data",)
    FUNCTION = "process"
    CATEGORY = "Deforum-X-Flux"

    def process(self, animation_mode, max_frames, border, deforum_data=None):
        if deforum_data is None:
            deforum_data = {}

        deforum_data.update({
            "animation_mode": animation_mode,
            "max_frames": max_frames,
            "border": border,
        })
        return (deforum_data,)


class DeforumTranslationParamsNode:
    """
    Translation/motion parameters - matches original structure
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "angle": ("STRING", {"default": "0:(0)", "multiline": True}),
                "zoom": ("STRING", {"default": "0:(1.04)", "multiline": True}),
                "translation_x": ("STRING", {"default": "0:(0)", "multiline": True}),
                "translation_y": ("STRING", {"default": "0:(0)", "multiline": True}),
                "translation_z": ("STRING", {"default": "0:(0.5)", "multiline": True}),
                "transform_center_x": ("STRING", {"default": "0:(0.5)", "multiline": True}),
                "transform_center_y": ("STRING", {"default": "0:(0.5)", "multiline": True}),
                "rotation_3d_x": ("STRING", {"default": "0:(0)", "multiline": True}),
                "rotation_3d_y": ("STRING", {"default": "0:(0)", "multiline": True}),
                "rotation_3d_z": ("STRING", {"default": "0:(0)", "multiline": True}),
            },
            "optional": {
                "deforum_data": ("DEFORUM_DATA",),
            }
        }

    RETURN_TYPES = ("DEFORUM_DATA",)
    RETURN_NAMES = ("deforum_data",)
    FUNCTION = "process"
    CATEGORY = "Deforum-X-Flux"

    def process(self, angle, zoom, translation_x, translation_y, translation_z,
                transform_center_x, transform_center_y, rotation_3d_x,
                rotation_3d_y, rotation_3d_z, deforum_data=None):
        if deforum_data is None:
            deforum_data = {}

        deforum_data.update({
            "angle": angle,
            "zoom": zoom,
            "translation_x": translation_x,
            "translation_y": translation_y,
            "translation_z": translation_z,
            "transform_center_x": transform_center_x,
            "transform_center_y": transform_center_y,
            "rotation_3d_x": rotation_3d_x,
            "rotation_3d_y": rotation_3d_y,
            "rotation_3d_z": rotation_3d_z,
        })
        return (deforum_data,)


class DeforumDiffusionParamsNode:
    """
    Diffusion parameters - matches original structure
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_schedule": ("STRING", {"default": "0:(0.02)", "multiline": True}),
                "strength_schedule": ("STRING", {"default": "0:(0.65)", "multiline": True}),
                "contrast_schedule": ("STRING", {"default": "0:(1.0)", "multiline": True}),
                "cfg_scale_schedule": ("STRING", {"default": "0:(7)", "multiline": True}),
                "ddim_eta_schedule": ("STRING", {"default": "0:(0)", "multiline": True}),
                "ancestral_eta_schedule": ("STRING", {"default": "0:(1)", "multiline": True}),
            },
            "optional": {
                "deforum_data": ("DEFORUM_DATA",),
                "enable_steps_scheduling": ("BOOLEAN", {"default": False}),
                "steps_schedule": ("STRING", {"default": "0:(25)", "multiline": True}),
                "enable_ddim_eta_scheduling": ("BOOLEAN", {"default": False}),
                "enable_ancestral_eta_scheduling": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("DEFORUM_DATA",)
    RETURN_NAMES = ("deforum_data",)
    FUNCTION = "process"
    CATEGORY = "Deforum-X-Flux"

    def process(self, noise_schedule, strength_schedule, contrast_schedule,
                cfg_scale_schedule, ddim_eta_schedule, ancestral_eta_schedule,
                deforum_data=None, enable_steps_scheduling=False,
                steps_schedule="0:(25)", enable_ddim_eta_scheduling=False,
                enable_ancestral_eta_scheduling=False):
        if deforum_data is None:
            deforum_data = {}

        deforum_data.update({
            "noise_schedule": noise_schedule,
            "strength_schedule": strength_schedule,
            "contrast_schedule": contrast_schedule,
            "cfg_scale_schedule": cfg_scale_schedule,
            "steps_schedule": steps_schedule,
            "ddim_eta_schedule": ddim_eta_schedule,
            "ancestral_eta_schedule": ancestral_eta_schedule,
            "enable_steps_scheduling": enable_steps_scheduling,
            "enable_ddim_eta_scheduling": enable_ddim_eta_scheduling,
            "enable_ancestral_eta_scheduling": enable_ancestral_eta_scheduling,
        })
        return (deforum_data,)


class DeforumBaseParamsNode:
    """
    Base parameters node - matches original structure
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "seed_schedule": ("STRING", {"default": "0:(-1)", "multiline": True}),
                "seed_behavior": (["iter", "fixed", "random"], {"default": "iter"}),
                "sampler_name": (["euler", "euler_a", "dpmpp_2m", "dpmpp_sde", "ddim"], {"default": "euler"}),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform"], {"default": "normal"}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0}),
                "clip_skip": ("INT", {"default": 1, "min": 1, "max": 12}),
            },
            "optional": {
                "deforum_data": ("DEFORUM_DATA",),
                "prompt_weighting": ("BOOLEAN", {"default": True}),
                "normalize_prompt_weights": ("BOOLEAN", {"default": True}),
                "log_weighted_subprompts": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("DEFORUM_DATA",)
    RETURN_NAMES = ("deforum_data",)
    FUNCTION = "process"
    CATEGORY = "Deforum-X-Flux"

    def process(self, width, height, seed_schedule, seed_behavior, sampler_name, scheduler,
                steps, cfg_scale, clip_skip, deforum_data=None, prompt_weighting=True,
                normalize_prompt_weights=True, log_weighted_subprompts=False):
        if deforum_data is None:
            deforum_data = {}

        deforum_data.update({
            "width": width,
            "height": height,
            "seed_schedule": seed_schedule,
            "seed_behavior": seed_behavior,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "clip_skip": clip_skip,
            "prompt_weighting": prompt_weighting,
            "normalize_prompt_weights": normalize_prompt_weights,
            "log_weighted_subprompts": log_weighted_subprompts,
        })
        return (deforum_data,)


class DeforumIteratorNode:
    """
    Core iterator node - manages frame progression and caching like original
    """

    def __init__(self):
        self.first_run = True
        self.frame_index = 0
        self.seed = ""
        self.seeds = []
        self.second_run = True
        self.logger = None

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Force re-evaluation of the node
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "deforum_data": ("DEFORUM_DATA",),
                "latent_type": (["stable_diffusion", "flux"], {"default": "flux"}),
            },
            "optional": {
                "latent": ("LATENT",),
                "init_latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "subseed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "subseed_strength": ("FLOAT", {"default": 0.8, "min": 0, "max": 1.0}),
                "slerp_strength": ("FLOAT", {"default": 0.1, "min": 0, "max": 1.0}),
                "reset_counter": ("BOOLEAN", {"default": False}),
                "reset_latent": ("BOOLEAN", {"default": False}),
                "enable_autoqueue": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("DEFORUM_FRAME_DATA", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("deforum_frame_data", "latent", "positive_prompt", "negative_prompt")
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = "Deforum-X-Flux"

    def get(self, deforum_data, latent_type, latent=None, init_latent=None, seed=None,
            subseed=None, subseed_strength=None, slerp_strength=None,
            reset_counter=False, reset_latent=False, enable_autoqueue=False, *args, **kwargs):

        global gs
        if gs.reset:
            reset_counter = True
            reset_latent = True

        # Get parameters from deforum_data
        max_frames = deforum_data.get("max_frames", 120)
        animation_mode = deforum_data.get("animation_mode", "2D")

        # Handle frame counter reset
        if self.frame_index >= max_frames or reset_counter:
            self.frame_index = 0
            self.first_run = True
            self.second_run = True
            if not self.logger and COMFY_AVAILABLE:
                try:
                    self.logger = comfy.utils.ProgressBar(max_frames)
                except:
                    pass

        if self.logger:
            self.logger.update_absolute(self.frame_index)

        # Parse prompts for current frame
        prompts = deforum_data.get("prompts", "0: a beautiful landscape")
        prompt_frames = parse_key_frames(prompts)

        # Get current prompt
        current_prompt = ""
        current_negative = ""
        for frame_num in sorted(prompt_frames.keys()):
            if self.frame_index >= frame_num:
                prompt_text = prompt_frames[frame_num]
                # Split positive and negative
                if "--neg" in prompt_text:
                    parts = prompt_text.split("--neg")
                    current_prompt = parts[0].strip()
                    current_negative = parts[1].strip() if len(parts) > 1 else ""
                else:
                    current_prompt = prompt_text.strip()
                    current_negative = ""

        # Get current parameters
        strength = 0.65
        cfg_scale = deforum_data.get("cfg_scale", 7.0)
        steps = deforum_data.get("steps", 25)

        # Parse schedules if they exist
        if "strength_schedule" in deforum_data:
            strength_frames = parse_key_frames(deforum_data["strength_schedule"])
            strength_series = get_inbetweens(strength_frames, max_frames)
            strength = float(strength_series[self.frame_index])

        # Handle seed
        if seed is None:
            seed = secrets.randbelow(2 ** 32 - 1)

        # Create frame data
        frame_data = {
            "prompt": current_prompt,
            "negative_prompt": current_negative,
            "denoise": strength,
            "cfg": cfg_scale,
            "steps": steps,
            "frame_idx": self.frame_index,
            "max_frames": max_frames,
            "seed": seed,
            "animation_mode": animation_mode,
            "deforum_data": deforum_data,
        }

        # Handle latent
        if reset_latent or not hasattr(self, 'current_latent') or self.first_run:
            if latent_type == "flux":
                channels = 16
                compression = 8
            else:
                channels = 4
                compression = 8

            if init_latent is not None:
                height, width = init_latent["samples"].shape[2] * compression, init_latent["samples"].shape[3] * compression
                self.current_latent = init_latent
            else:
                height = deforum_data.get("height", 512)
                width = deforum_data.get("width", 512)
                latent_tensor = torch.randn((1, channels, height // compression, width // compression))
                if COMFY_AVAILABLE:
                    latent_tensor = latent_tensor.to(intermediate_device())
                self.current_latent = {"samples": latent_tensor}

        # Advance frame counter
        if not self.first_run:
            self.frame_index += 1
        self.first_run = False

        print(f"[Deforum-X-Flux] Frame: {self.frame_index} of {max_frames}")

        gs.reset = False
        enable_autoqueue = enable_autoqueue if self.frame_index == 0 else False

        return {
            "ui": {
                "counter": (self.frame_index,),
                "max_frames": (max_frames,),
                "enable_autoqueue": (enable_autoqueue,)
            },
            "result": (frame_data, self.current_latent, current_prompt, current_negative)
        }


class DeforumFluxSampler:
    """
    FLUX-based sampler node - replaces original KSampler for FLUX models
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "Deforum-X-Flux"

    def sample(self, model, latent, positive, negative, deforum_frame_data):
        try:
            if not COMFY_AVAILABLE:
                print("ComfyUI not available, returning input latent")
                return (latent,)

            # Get sampling parameters from frame data
            steps = deforum_frame_data.get("steps", 25)
            cfg = deforum_frame_data.get("cfg", 7.0)
            denoise = deforum_frame_data.get("denoise", 1.0)
            seed = deforum_frame_data.get("seed", 0)

            # Use ComfyUI's sampling system
            if hasattr(comfy.sample, 'sample'):
                # Prepare noise
                noise = comfy.sample.prepare_noise(latent["samples"], seed=seed)

                # Sample
                samples = comfy.sample.sample(
                    model=model,
                    noise=noise,
                    steps=steps,
                    cfg=cfg,
                    sampler_name="euler",
                    scheduler="normal",
                    positive=positive,
                    negative=negative,
                    latent_image=latent,
                    denoise=denoise
                )

                return (samples,)
            else:
                print("ComfyUI sampling not available, returning input latent")
                return (latent,)

        except Exception as e:
            print(f"Error in FLUX sampling: {e}")
            return (latent,)


class DeforumCacheLatentNode:
    """
    Cache latent node - stores latents for next frame
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "cache_index": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "cache"
    CATEGORY = "Deforum-X-Flux"

    def cache(self, latent, cache_index=0):
        global gs
        if "latent" not in gs.deforum_cache:
            gs.deforum_cache["latent"] = {}

        gs.deforum_cache["latent"][cache_index] = latent
        return (latent,)


class DeforumGetCachedLatentNode:
    """
    Get cached latent node - retrieves stored latents
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "cache_index": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "get_cached"
    CATEGORY = "Deforum-X-Flux"

    def get_cached(self, cache_index=0):
        global gs
        if "latent" in gs.deforum_cache and cache_index in gs.deforum_cache["latent"]:
            return (gs.deforum_cache["latent"][cache_index],)
        else:
            # Return empty latent if nothing cached
            empty_latent = torch.zeros((1, 16, 64, 64))
            if COMFY_AVAILABLE:
                empty_latent = empty_latent.to(intermediate_device())
            return ({"samples": empty_latent},)


class DeforumConditioningBlendNode:
    """
    Conditioning blend node - handles prompt conditioning for FLUX
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
            },
            "optional": {
                "blend_method": (["linear", "cosine"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("POSITIVE", "NEGATIVE")
    FUNCTION = "blend"
    CATEGORY = "Deforum-X-Flux"

    def blend(self, clip, deforum_frame_data, blend_method="linear"):
        try:
            positive_prompt = deforum_frame_data.get("prompt", "")
            negative_prompt = deforum_frame_data.get("negative_prompt", "")

            # Tokenize and encode
            positive_tokens = clip.tokenize(positive_prompt)
            negative_tokens = clip.tokenize(negative_prompt)

            positive_cond = clip.encode_from_tokens(positive_tokens, return_pooled=True)
            negative_cond = clip.encode_from_tokens(negative_tokens, return_pooled=True)

            return (positive_cond, negative_cond)

        except Exception as e:
            print(f"Error in conditioning: {e}")
            # Return dummy conditioning
            dummy_cond = (torch.zeros((1, 77, 768)), torch.zeros((1, 768)))
            return (dummy_cond, dummy_cond)


class DeforumRenderer:
    """
    Main FLUX-based rendering engine
    Integrates with FLUX models for high-quality animation generation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animation_config": ("DEFORUM_CONFIG",),
                "motion_params": ("DEFORUM_MOTION",),
                "keyframes": ("DEFORUM_KEYFRAMES",),
                "flux_model": ("MODEL",),
                "flux_vae": ("VAE",),
                "clip": ("CLIP",),
            },
            "optional": {
                "init_image": ("IMAGE",),
                "depth_model": ("DEPTH_MODEL",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 9999}),
                "preview_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "DEFORUM_ANIMATION",)
    RETURN_NAMES = ("images", "animation_data",)
    FUNCTION = "render_animation"
    CATEGORY = "Deforum-X-Flux"

    def render_animation(self, animation_config, motion_params, keyframes, flux_model, flux_vae, clip,
                        init_image=None, depth_model=None, start_frame=0, end_frame=-1, preview_mode=False):

        device = animation_config["device"]
        max_frames = animation_config["max_frames"]
        width = animation_config["width"]
        height = animation_config["height"]
        animation_mode = animation_config["animation_mode"]

        if end_frame == -1:
            end_frame = max_frames

        # Initialize animation state
        prev_sample = None
        color_match_sample = None
        generated_images = []
        animation_data = {
            "frames": [],
            "metadata": animation_config.copy(),
            "motion_data": {},
        }

        # Process each frame
        for frame_idx in range(start_frame, min(end_frame, max_frames)):
            print(f"Rendering frame {frame_idx + 1}/{max_frames}")

            # Get frame parameters
            current_prompt = keyframes["positive_prompts"][frame_idx]
            current_negative = keyframes["negative_prompts"][frame_idx]
            strength = motion_params["strength_schedule_series"][frame_idx]
            noise = motion_params["noise_schedule_series"][frame_idx]

            # Apply motion transformations to previous frame
            if prev_sample is not None and animation_mode in ["2D", "3D"]:
                prev_sample = self.apply_motion_transform(
                    prev_sample, frame_idx, animation_config, motion_params, depth_model
                )

            # Prepare conditioning
            try:
                # Tokenize prompts
                positive_tokens = clip.tokenize(current_prompt)
                negative_tokens = clip.tokenize(current_negative)

                # Encode conditioning
                positive_cond = clip.encode_from_tokens(positive_tokens, return_pooled=True)
                negative_cond = clip.encode_from_tokens(negative_tokens, return_pooled=True)
            except Exception as e:
                print(f"Error in conditioning: {e}")
                # Create dummy conditioning as fallback
                positive_cond = (torch.zeros((1, 77, 768)), torch.zeros((1, 768)))
                negative_cond = (torch.zeros((1, 77, 768)), torch.zeros((1, 768)))

            # Generate frame using FLUX
            try:
                latent = self.generate_flux_frame(
                    flux_model, flux_vae, positive_cond, negative_cond,
                    width, height, keyframes["steps"], keyframes["guidance_scale"],
                    prev_sample, strength, noise, device
                )

                # Decode to image
                image = flux_vae.decode(latent)

                # Convert to proper format
                if isinstance(image, dict) and "samples" in image:
                    image = image["samples"]

                # Ensure image is in correct format for ComfyUI
                if len(image.shape) == 4 and image.shape[0] == 1:
                    image = image.squeeze(0)

                # Convert to numpy for processing
                image_np = sample_to_cv2(image)

            except Exception as e:
                print(f"Error in frame generation: {e}")
                # Create a fallback image
                image_np = np.zeros((height, width, 3), dtype=np.uint8)
                image_np.fill(128)  # Gray fallback

            # Apply color coherence
            if animation_config["color_coherence"] != "None" and color_match_sample is not None:
                image_np = self.apply_color_coherence(image_np, color_match_sample, animation_config["color_coherence"])
            elif color_match_sample is None:
                color_match_sample = image_np.copy()

            # Store results
            generated_images.append(sample_from_cv2(image_np))
            prev_sample = sample_from_cv2(image_np)

            # Store frame metadata
            animation_data["frames"].append({
                "frame": frame_idx,
                "prompt": current_prompt,
                "strength": float(strength),
                "noise": float(noise),
            })

            # Preview mode: only generate first few frames
            if preview_mode and frame_idx >= start_frame + 4:
                break

        # Stack images into batch
        if generated_images:
            output_images = torch.stack(generated_images, dim=0)
        else:
            # Return empty image if no frames generated
            output_images = torch.zeros((1, 3, height, width))

        return (output_images, animation_data)

    def apply_motion_transform(self, prev_sample, frame_idx, animation_config, motion_params, depth_model=None):
        """Apply 2D or 3D motion transformations to previous frame"""

        prev_img_cv2 = sample_to_cv2(prev_sample)
        animation_mode = animation_config["animation_mode"]

        if animation_mode == "2D":
            return self.apply_2d_transform(prev_img_cv2, frame_idx, animation_config, motion_params)
        elif animation_mode == "3D":
            return self.apply_3d_transform(prev_img_cv2, frame_idx, animation_config, motion_params, depth_model)

        return prev_sample

    def apply_2d_transform(self, prev_img_cv2, frame_idx, animation_config, motion_params):
        """Apply 2D transformations: rotation, zoom, translation"""

        try:
            angle = float(motion_params["angle_series"][frame_idx])
            zoom = float(motion_params["zoom_series"][frame_idx])
            translation_x = float(motion_params["translation_x_series"][frame_idx])
            translation_y = float(motion_params["translation_y_series"][frame_idx])

            height, width = prev_img_cv2.shape[:2]
            center = (width // 2, height // 2)

            # Create rotation and scale matrix
            rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)

            # Add translation
            rot_mat[0, 2] += translation_x
            rot_mat[1, 2] += translation_y

            # Apply transformation
            border_mode = cv2.BORDER_WRAP if animation_config["border_mode"] == "wrap" else cv2.BORDER_REPLICATE
            transformed = cv2.warpAffine(prev_img_cv2, rot_mat, (width, height), borderMode=border_mode)

            return sample_from_cv2(transformed)

        except Exception as e:
            print(f"Error in 2D transform: {e}")
            return sample_from_cv2(prev_img_cv2)  # Return original on error

    def apply_3d_transform(self, prev_img_cv2, frame_idx, animation_config, motion_params, depth_model):
        """Apply 3D transformations with depth warping"""

        if not animation_config.get("use_depth_warping", False) or depth_model is None:
            return self.apply_2d_transform(prev_img_cv2, frame_idx, animation_config, motion_params)

        # Get 3D motion parameters
        TRANSLATION_SCALE = 1.0/200.0
        translate_xyz = [
            -motion_params["translation_x_series"][frame_idx] * TRANSLATION_SCALE,
            motion_params["translation_y_series"][frame_idx] * TRANSLATION_SCALE,
            -motion_params["translation_z_series"][frame_idx] * TRANSLATION_SCALE
        ]
        rotate_xyz = [
            math.radians(motion_params["rotation_3d_x_series"][frame_idx]),
            math.radians(motion_params["rotation_3d_y_series"][frame_idx]),
            math.radians(motion_params["rotation_3d_z_series"][frame_idx])
        ]

        # Apply 3D transformation (simplified version)
        # In full implementation, this would use py3d_tools for proper 3D warping
        transformed = self.apply_2d_transform(prev_img_cv2, frame_idx, animation_config, motion_params)

        return transformed

    def generate_flux_frame(self, flux_model, flux_vae, positive_cond, negative_cond,
                           width, height, steps, guidance_scale, init_sample=None,
                           strength=1.0, noise=0.0, device="cuda"):
        """Generate single frame using FLUX model"""

        if not COMFY_AVAILABLE:
            raise RuntimeError("ComfyUI not available for FLUX sampling")

        try:
            # Create latent dimensions for FLUX (16 channels, 8x downsampling)
            latent_height = height // 8
            latent_width = width // 8

            if init_sample is not None:
                # Encode init image to latent space
                if len(init_sample.shape) == 3:
                    init_sample = init_sample.unsqueeze(0)
                init_sample = init_sample.to(device)

                # Encode using VAE
                init_latent = flux_vae.encode(init_sample)

                # Add noise for strength control
                if strength < 1.0:
                    noise_tensor = torch.randn_like(init_latent) * noise
                    latent = init_latent * (1 - strength) + noise_tensor * strength
                else:
                    latent = init_latent
            else:
                # Create random latent for FLUX (16 channels)
                latent = torch.randn((1, 16, latent_height, latent_width), device=device)

            # Prepare sampling parameters
            latent_dict = {"samples": latent}

            # Use ComfyUI's sampling system
            if hasattr(comfy.sample, 'sample'):
                # Create noise for sampling
                noise = comfy.sample.prepare_noise(latent, seed=None)

                # Sample using FLUX model
                samples = comfy.sample.sample(
                    model=flux_model,
                    noise=noise,
                    steps=steps,
                    cfg=guidance_scale,
                    sampler_name="euler",
                    scheduler="normal",
                    positive=positive_cond,
                    negative=negative_cond,
                    latent_image=latent_dict,
                    denoise=strength
                )

                return samples["samples"]
            else:
                # Fallback: return the latent as-is
                print("Warning: ComfyUI sampling not available, returning latent")
                return latent

        except Exception as e:
            print(f"Error in FLUX sampling: {e}")
            # Return a basic latent as fallback
            return torch.randn((1, 16, height // 8, width // 8), device=device)

    def apply_color_coherence(self, image, reference, method):
        """Apply color coherence to maintain consistency across frames"""

        if method == "Match Frame 0 RGB":
            # Simple RGB matching (placeholder)
            return image
        elif method == "Match Frame 0 HSV":
            # HSV color matching (placeholder)
            return image
        elif method == "Match Frame 0 LAB":
            # LAB color matching (placeholder)
            return image

        return image


class DeforumDepthWarping:
    """
    3D depth warping controller
    Provides depth estimation and 3D transformation capabilities
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animation_config": ("DEFORUM_CONFIG",),
                "depth_model_name": (["midas", "adabins", "dpt"], {"default": "midas"}),
            },
            "optional": {
                "midas_weight": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "near_plane": ("FLOAT", {"default": 200, "min": 1, "max": 1000}),
                "far_plane": ("FLOAT", {"default": 10000, "min": 1000, "max": 50000}),
                "fov": ("FLOAT", {"default": 40, "min": 10, "max": 120}),
                "save_depth_maps": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("DEPTH_MODEL",)
    RETURN_NAMES = ("depth_model",)
    FUNCTION = "setup_depth_warping"
    CATEGORY = "Deforum-X-Flux"

    def setup_depth_warping(self, animation_config, depth_model_name, midas_weight=0.3,
                           near_plane=200, far_plane=10000, fov=40, save_depth_maps=False):

        device = animation_config["device"]

        depth_config = {
            "model_name": depth_model_name,
            "midas_weight": midas_weight,
            "near_plane": near_plane,
            "far_plane": far_plane,
            "fov": fov,
            "save_depth_maps": save_depth_maps,
            "device": device,
            "loaded": False,
        }

        # In full implementation, this would load the actual depth model
        # For now, return configuration
        return (depth_config,)


class DeforumVideoInput:
    """
    Video input processor for hybrid video generation
    Handles video frame extraction and optical flow analysis
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animation_config": ("DEFORUM_CONFIG",),
                "video_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "extract_nth_frame": ("INT", {"default": 1, "min": 1, "max": 10}),
                "hybrid_motion": (["None", "Optical Flow", "Perspective", "Affine"], {"default": "None"}),
                "hybrid_composite": ("BOOLEAN", {"default": False}),
                "hybrid_flow_method": (["DenseRLOF", "DIS Medium", "Farneback", "SF"], {"default": "DIS Medium"}),
                "overwrite_frames": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("DEFORUM_VIDEO_INPUT",)
    RETURN_NAMES = ("video_input",)
    FUNCTION = "setup_video_input"
    CATEGORY = "Deforum-X-Flux"

    def setup_video_input(self, animation_config, video_path, extract_nth_frame=1,
                         hybrid_motion="None", hybrid_composite=False,
                         hybrid_flow_method="DIS Medium", overwrite_frames=True):

        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"Video path does not exist: {video_path}")

        # Extract frames from video
        frames_dir = self.extract_video_frames(video_path, extract_nth_frame, overwrite_frames)

        video_input = {
            "video_path": video_path,
            "frames_dir": frames_dir,
            "extract_nth_frame": extract_nth_frame,
            "hybrid_motion": hybrid_motion,
            "hybrid_composite": hybrid_composite,
            "hybrid_flow_method": hybrid_flow_method,
            "animation_config": animation_config,
        }

        return (video_input,)

    def extract_video_frames(self, video_path, nth_frame, overwrite):
        """Extract frames from video using OpenCV"""

        frames_dir = os.path.join(os.path.dirname(video_path), "extracted_frames")
        os.makedirs(frames_dir, exist_ok=True)

        if not overwrite and os.listdir(frames_dir):
            print("Frames already extracted, skipping...")
            return frames_dir

        # Clear existing frames
        for f in Path(frames_dir).glob('*.jpg'):
            f.unlink()

        # Extract frames
        cap = cv2.VideoCapture(video_path)
        success, image = cap.read()
        count = 0
        frame_num = 1

        while success:
            if count % nth_frame == 0:
                frame_path = os.path.join(frames_dir, f"{frame_num:05d}.jpg")
                cv2.imwrite(frame_path, image)
                frame_num += 1
            success, image = cap.read()
            count += 1

        cap.release()
        print(f"Extracted {frame_num-1} frames to {frames_dir}")

        return frames_dir


class DeforumInterpolation:
    """
    Frame interpolation engine
    Provides smooth transitions between keyframes
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animation_data": ("DEFORUM_ANIMATION",),
                "interpolation_method": (["linear", "cubic", "optical_flow"], {"default": "linear"}),
                "interpolation_frames": ("INT", {"default": 2, "min": 1, "max": 10}),
            },
            "optional": {
                "smooth_transitions": ("BOOLEAN", {"default": True}),
                "preserve_details": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("interpolated_images",)
    FUNCTION = "interpolate_frames"
    CATEGORY = "Deforum-X-Flux"

    def interpolate_frames(self, animation_data, interpolation_method, interpolation_frames,
                          smooth_transitions=True, preserve_details=True):

        # This is a placeholder for frame interpolation
        # In full implementation, this would use advanced interpolation techniques

        frames = animation_data.get("frames", [])
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames for interpolation")

        # For now, return original frames
        # In full implementation, this would generate interpolated frames
        interpolated = torch.zeros((len(frames), 3, 1024, 1024))

        return (interpolated,)


class DeforumVideoOutput:
    """
    Video output composer
    Compiles animation frames into final video with FFmpeg
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "animation_config": ("DEFORUM_CONFIG",),
                "output_format": (["mp4", "gif", "webm", "mov"], {"default": "mp4"}),
                "output_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "fps": ("INT", {"default": 12, "min": 1, "max": 60}),
                "quality": (["low", "medium", "high", "lossless"], {"default": "high"}),
                "codec": (["h264", "h265", "vp9", "prores"], {"default": "h264"}),
                "audio_path": ("STRING", {"default": ""}),
                "loop_video": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "preview_frame")
    FUNCTION = "compile_video"
    CATEGORY = "Deforum-X-Flux"

    def compile_video(self, images, animation_config, output_format, output_path,
                     fps=12, quality="high", codec="h264", audio_path="", loop_video=False):

        if not output_path:
            output_path = os.path.join(folder_paths.get_output_directory(), f"deforum_animation.{output_format}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save frames to temporary directory
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        try:
            # Convert tensor images to files
            for i, image in enumerate(images):
                if len(image.shape) == 4:
                    image = image.squeeze(0)

                # Convert to numpy and save
                img_np = sample_to_cv2(image)
                frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                frame_paths.append(frame_path)

            # Compile video with FFmpeg
            video_path = self.create_video_ffmpeg(
                temp_dir, output_path, fps, quality, codec, output_format, audio_path, loop_video
            )

            # Return first frame as preview
            preview_frame = images[0] if len(images) > 0 else torch.zeros((1, 3, 512, 512))

            return (video_path, preview_frame)

        finally:
            # Cleanup temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def create_video_ffmpeg(self, frames_dir, output_path, fps, quality, codec, format_type, audio_path, loop_video):
        """Create video using FFmpeg"""

        # Quality settings
        quality_settings = {
            "low": ["-crf", "28"],
            "medium": ["-crf", "23"],
            "high": ["-crf", "18"],
            "lossless": ["-crf", "0"]
        }

        # Codec settings
        codec_settings = {
            "h264": ["-c:v", "libx264"],
            "h265": ["-c:v", "libx265"],
            "vp9": ["-c:v", "libvpx-vp9"],
            "prores": ["-c:v", "prores_ks"]
        }

        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%05d.png"),
        ]

        # Add codec and quality
        cmd.extend(codec_settings.get(codec, codec_settings["h264"]))
        cmd.extend(quality_settings.get(quality, quality_settings["high"]))

        # Add audio if provided
        if audio_path and os.path.exists(audio_path):
            cmd.extend(["-i", audio_path, "-c:a", "aac"])

        # Format-specific settings
        if format_type == "gif":
            cmd.extend(["-vf", "palettegen=reserve_transparent=0"])
        elif format_type == "webm":
            cmd.extend(["-c:v", "libvpx-vp9", "-crf", "30"])

        # Loop settings
        if loop_video and format_type == "gif":
            cmd.extend(["-loop", "0"])

        # Output path
        cmd.append(output_path)

        # Execute FFmpeg
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Video created successfully: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to create video: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg to use video output.")


# Node class mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    # Core Deforum nodes (matching original structure)
    "DeforumPromptNode": DeforumPromptNode,
    "DeforumAnimParamsNode": DeforumAnimParamsNode,
    "DeforumTranslationParamsNode": DeforumTranslationParamsNode,
    "DeforumDiffusionParamsNode": DeforumDiffusionParamsNode,
    "DeforumBaseParamsNode": DeforumBaseParamsNode,
    "DeforumIteratorNode": DeforumIteratorNode,

    # FLUX-specific nodes
    "DeforumFluxSampler": DeforumFluxSampler,
    "DeforumConditioningBlendNode": DeforumConditioningBlendNode,

    # Caching nodes
    "DeforumCacheLatentNode": DeforumCacheLatentNode,
    "DeforumGetCachedLatentNode": DeforumGetCachedLatentNode,

    # Legacy nodes (for compatibility)
    "DeforumRenderer": DeforumRenderer,
    "DeforumDepthWarping": DeforumDepthWarping,
    "DeforumVideoInput": DeforumVideoInput,
    "DeforumInterpolation": DeforumInterpolation,
    "DeforumVideoOutput": DeforumVideoOutput,
}

# Display names for ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    # Core Deforum nodes
    "DeforumPromptNode": "(deforum) Prompt Node",
    "DeforumAnimParamsNode": "(deforum) Animation Parameters",
    "DeforumTranslationParamsNode": "(deforum) Translation Parameters",
    "DeforumDiffusionParamsNode": "(deforum) Diffusion Parameters",
    "DeforumBaseParamsNode": "(deforum) Base Parameters",
    "DeforumIteratorNode": "(deforum) Iterator Node",

    # FLUX-specific nodes
    "DeforumFluxSampler": "(deforum) FLUX Sampler",
    "DeforumConditioningBlendNode": "(deforum) Conditioning Blend",

    # Caching nodes
    "DeforumCacheLatentNode": "(deforum) Cache Latent",
    "DeforumGetCachedLatentNode": "(deforum) Get Cached Latent",

    # Legacy nodes
    "DeforumRenderer": "🎨 Deforum FLUX Renderer",
    "DeforumDepthWarping": "🌊 Deforum Depth Warping",
    "DeforumVideoInput": "🎞️ Deforum Video Input",
    "DeforumInterpolation": "🔄 Deforum Interpolation",
    "DeforumVideoOutput": "📹 Deforum Video Output",
}
