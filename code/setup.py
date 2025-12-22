"""
Setup configuration for Options Backtesting System

Install in development mode:
    pip install -e .

Install for production:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent.parent / 'README.md'
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

setup(
    name="options-backtester",
    version="1.0.0",
    description="Professional Options Strategy Backtesting Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Options Backtester Team",
    author_email="info@optionsbacktester.com",
    url="https://github.com/yourusername/options-backtester",
    license="MIT",

    packages=find_packages(exclude=["tests", "tests.*"]),

    # Core dependencies
    install_requires=[
        "numpy>=2.0.2",
        "pandas>=2.3.3",
        "scipy>=1.13.1",
        "matplotlib>=3.9.4",
        "plotly>=6.5.0",
        "seaborn>=0.13.2",
        "doltpy>=2.0.0",
    ],

    # Development dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },

    # Python version requirement
    python_requires=">=3.9",

    # Package classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],

    # Keywords for package discovery
    keywords="options trading backtesting finance derivatives quantitative",

    # Project URLs
    project_urls={
        "Documentation": "https://github.com/yourusername/options-backtester/docs",
        "Source": "https://github.com/yourusername/options-backtester",
        "Tracker": "https://github.com/yourusername/options-backtester/issues",
    },

    # Include package data
    include_package_data=True,
    package_data={
        "backtester": ["py.typed"],  # PEP 561 type hint marker
    },

    # Entry points (if needed for CLI tools)
    entry_points={
        "console_scripts": [
            # "backtest-cli=backtester.cli:main",  # Uncomment if you add CLI
        ],
    },

    # Zip safe flag
    zip_safe=False,
)
