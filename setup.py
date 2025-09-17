#!/usr/bin/env python
"""
Setup script for Smart Grid Load Balancing Optimizer
===================================================

Install with:
    pip install -e .

For development:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
import re
from pathlib import Path

# Read version from a version file or main module
def get_version() -> str:
    """Extract version from main module."""
    try:
        with open("main.py", "r", encoding="utf-8") as f:
            content = f.read()
            # Look for __version__ = "x.y.z" pattern
            version_match = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content)
            if version_match:
                return version_match.group(1)
    except FileNotFoundError:
        pass
    return "1.0.0"  # Default version

# Read long description from README
def get_long_description() -> str:
    """Get long description from README.md."""
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def get_requirements() -> list:
    """Get requirements from requirements.txt."""
    requirements = []
    req_path = Path("requirements.txt")
    
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith("#"):
                    # Remove inline comments
                    requirement = line.split("#")[0].strip()
                    if requirement:
                        requirements.append(requirement)
    
    return requirements

setup(
    name="smart-grid-optimizer",
    version=get_version(),
    description="AI-powered smart grid load balancing and anomaly detection system",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Smart Grid Team",
    author_email="team@smartgrid.ai",
    url="https://github.com/your-username/smart-grid-optimizer",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/smart-grid-optimizer/issues",
        "Documentation": "https://github.com/your-username/smart-grid-optimizer#readme",
        "Source Code": "https://github.com/your-username/smart-grid-optimizer",
    },
    packages=find_packages(),
    py_modules=[
        "main",
        "integrated_smart_grid_pipeline", 
        "smart_grid_data_generator",
        "smart_grid_dashboard",
        "backtest_engine",
        "schema_validator",
        "simplified_config",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0", 
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pytest-cov>=4.1.0",
            "pre-commit>=3.4.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",  # GPU acceleration
            "nvidia-ml-py>=12.0.0",  # GPU monitoring
        ]
    },
    entry_points={
        "console_scripts": [
            "smart-grid=main:main",
            "smart-grid-train=main:train_cmd",
            "smart-grid-dashboard=main:dashboard_cmd",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
        "Topic :: Utilities",
    ],
    keywords=[
        "smart-grid", "machine-learning", "energy", "forecasting", 
        "anomaly-detection", "time-series", "gpu-acceleration",
        "dashboard", "monitoring", "ai", "catboost", "pytorch"
    ],
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
)