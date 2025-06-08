"""Setup script for FluxMD package."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version
exec(open("fluxmd/__version__.py").read())

setup(
    name="fluxmd",
    version=__version__,
    author="Myunghyun Jeong",
    author_email="mhjonathan@gm.gist.ac.kr",
    description="Energy Flux Differential Analysis for Protein-Ligand Binding Sites",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/panelarin/FluxMD",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fluxmd=fluxmd.cli:main",
            "fluxmd-uma=fluxmd.cli:main_uma",
            "fluxmd-uma-interactive=fluxmd_uma_interactive:main",
            "fluxmd-dna=fluxmd.utils.dna_to_pdb:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fluxmd": ["data/*.json", "data/*.csv"],
    },
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "sphinx>=6.0",
            "sphinx-rtd-theme>=1.3",
        ],
        "gpu": [
            "torch>=2.0.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.12.0",
        ],
    },
)