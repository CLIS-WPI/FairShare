"""
Setup script for Fuzzy-Fairness DSS LEO package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="fuzzy-fairness-dss-leo",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fuzzy-Fairness Dynamic Spectrum Sharing for LEO Satellite Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/fuzzy-fairness-dss-leo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Communications :: Telephony",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black",
            "flake8",
            "isort",
            "bandit",
        ],
        "gpu": [
            "tensorflow>=2.16.0",
            "sionna>=1.2.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "fuzzy-dss=src.main:main",
        ],
    },
)

