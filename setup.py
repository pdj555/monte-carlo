"""Setup configuration for Monte Carlo stock price simulation package."""

from pathlib import Path
from setuptools import setup, find_packages

# Read the README for the long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="monte-carlo-stock-sim",
    version="1.0.0",
    author="Monte Carlo Stock Simulation Contributors",
    author_email="",
    description="Monte Carlo stock price simulation with multiple models and advanced analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pdj555/monte-carlo",
    project_urls={
        "Bug Tracker": "https://github.com/pdj555/monte-carlo/issues",
        "Documentation": "https://github.com/pdj555/monte-carlo/blob/main/README.md",
        "Source Code": "https://github.com/pdj555/monte-carlo",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="monte-carlo simulation finance stocks trading risk-analysis gbm",
    py_modules=[
        "MonteCarlo",
        "analysis",
        "cli",
        "data",
        "simulation",
        "viz",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=4.0",
            "flake8>=6.0",
            "black>=23.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "monte-carlo=cli:main",
            "monte-carlo-sim=MonteCarlo:main",
        ],
    },
)
