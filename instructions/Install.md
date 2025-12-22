# Installation Guide

## Quick Install (Recommended)

Using `uv` for fast package management:

```bash
cd "Options Backtester"
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Manual Install

If you don't have `uv`, use standard pip:

```bash
cd "Options Backtester"
source .venv/bin/activate
pip install -r requirements.txt
```

## Core Dependencies

### Required Packages

- **numpy** (2.0.2+): Numerical computing
- **pandas** (2.3.3+): Data manipulation
- **scipy** (1.13.1+): Scientific computing, statistics
- **matplotlib** (3.9.4+): Plotting
- **plotly** (6.5.0+): Interactive visualizations
- **seaborn** (0.13.2+): Statistical visualizations
- **yfinance** (0.2.66+): Yahoo Finance data
- **numba**: JIT compilation (optional, for performance)
- **statsmodels**: Advanced statistics
- **pyyaml**: Configuration files
- **pytest**: Testing
- **jupyter**: Notebook support
- **notebook**: Jupyter notebooks

## Verify Installation

Run this to verify all packages are installed correctly:

```bash
cd "Options Backtester"
source .venv/bin/activate
python -c "
from backtester import (
    DoltHubAdapter,
    MarketDataLoader,
    PerformanceMetrics,
    BacktestEngine
)
print('✅ All packages installed correctly!')
"
```

## Additional Setup

### Dolt Database (for DoltHub integration)

Install Dolt for accessing the options dataset:

**macOS:**
```bash
brew install dolt
```

**Linux:**
```bash
curl -L https://github.com/dolthub/dolt/releases/latest/download/install.sh | sudo bash
```

**Windows:**
Download from https://github.com/dolthub/dolt/releases

### Clone Options Dataset

```bash
cd ~/Desktop
mkdir dolt_data && cd dolt_data
dolt clone post-no-preference/options
```

This downloads ~several GB of historical options data. Use `--depth 1` for a shallow clone if space is limited.

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Reinstall specific package
uv pip install --force-reinstall package_name

# Or reinstall all
uv pip install --force-reinstall -r requirements.txt
```

### Missing Packages

If a package is missing:

```bash
uv pip install package_name
```

### Virtual Environment Issues

If the virtual environment is broken:

```bash
# Remove old environment
rm -rf .venv

# Create new one
uv venv

# Activate
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install packages
uv pip install -r requirements.txt
```

### Jupyter Kernel Issues

If Jupyter can't find the packages:

```bash
# Install ipykernel
uv pip install ipykernel

# Register kernel
python -m ipykernel install --user --name=options-backtester --display-name="Options Backtester"

# Use this kernel in Jupyter
```

## Package Versions

The backtester has been tested with:

- Python 3.9+
- NumPy 2.0.2
- Pandas 2.3.3
- SciPy 1.13.1
- Matplotlib 3.9.4
- Plotly 6.5.0
- Seaborn 0.13.2
- yfinance 0.2.66

Newer versions should work, but if you encounter issues, try pinning to these versions.

## Performance Optimization (Optional)

For faster backtests, install additional packages:

```bash
# Numba for JIT compilation
uv pip install numba

# Faster CSV parsing
uv pip install python-rapidjson

# Parallel processing
uv pip install joblib
```

## Development Setup (Optional)

For development work:

```bash
# Install development dependencies
uv pip install black flake8 mypy pylint

# Install in editable mode
uv pip install -e .
```

## Platform-Specific Notes

### macOS

- Requires Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for system dependencies: `brew install`

### Linux

- May need build tools: `sudo apt-get install build-essential`
- Some packages need: `sudo apt-get install python3-dev`

### Windows

- Requires Microsoft C++ Build Tools
- Use WSL2 for better compatibility
- Or install via Anaconda for easier setup

## Minimal Install

For basic backtesting without Jupyter or DoltHub:

```bash
uv pip install numpy pandas scipy matplotlib yfinance
```

## Docker (Alternative)

If you prefer Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python"]
```

Build and run:
```bash
docker build -t options-backtester .
docker run -it options-backtester
```

## Verification Checklist

After installation, verify:

- [ ] Python 3.9+ installed
- [ ] Virtual environment activated
- [ ] All packages installed (`uv pip list`)
- [ ] Backtester imports work
- [ ] Jupyter notebooks launch
- [ ] Dolt installed (for DoltHub data)
- [ ] Options dataset cloned (optional)

## Getting Help

If you encounter issues:

1. Check this guide
2. Review error messages
3. Try reinstalling packages
4. Check GitHub issues
5. Verify Python version compatibility

## License

MIT License - See LICENSE file for details.
