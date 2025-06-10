#!/usr/bin/env python3
"""
Setup script for ComfyUI Deforum-X-Flux Nodes
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="comfyui-deforum-x-flux-nodes",
    version="1.0.0",
    author="ComfyUI Deforum-X-Flux Team",
    author_email="",
    description="Professional video animation nodes for ComfyUI with FLUX model integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/comfyui-deforum-x-flux-nodes",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "video": [
            "ffmpeg-python>=0.2.0",
        ],
        "3d": [
            "pytorch3d>=0.7.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "*.md",
            "*.txt",
            "*.json",
            "workflows/*.json",
            "web/css/*.css",
            "web/js/*.js",
        ],
    },
    entry_points={
        "console_scripts": [
            "deforum-test=test_nodes_simple:main",
        ],
    },
    keywords=[
        "comfyui",
        "deforum", 
        "flux",
        "animation",
        "video",
        "ai",
        "machine-learning",
        "stable-diffusion",
        "image-generation",
        "video-generation",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/comfyui-deforum-x-flux-nodes/issues",
        "Source": "https://github.com/your-username/comfyui-deforum-x-flux-nodes",
        "Documentation": "https://github.com/your-username/comfyui-deforum-x-flux-nodes/blob/main/README.md",
        "Original Deforum": "https://github.com/XmYx/deforum-comfy-nodes",
    },
)
