# UV Package Manager Integration
# This file provides UV-based package management when USE_UV=true

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"Success: {description}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {description}")
        print(f"Command failed: {e}")
        sys.exit(1)

def install_dependencies():
    """Install dependencies using UV if enabled, otherwise pip."""
    if os.getenv("USE_UV", "false").lower() == "true":
        print("Using UV package manager...")
        run_command("uv sync", "Sync dependencies with UV")
        run_command("uv pip install -e .", "Install package in development mode")
    else:
        print("Using pip package manager...")
        run_command("pip install -e .", "Install package in development mode")

def install_dev_dependencies():
    """Install development dependencies."""
    if os.getenv("USE_UV", "false").lower() == "true":
        run_command("uv pip install -e .[dev]", "Install dev dependencies with UV")
    else:
        run_command("pip install -e .[dev]", "Install dev dependencies with pip")

def update_dependencies():
    """Update dependencies."""
    if os.getenv("USE_UV", "false").lower() == "true":
        run_command("uv lock --upgrade", "Update UV lock file")
        run_command("uv sync", "Sync updated dependencies")
    else:
        run_command("pip install --upgrade pip setuptools wheel", "Update pip and tools")
        run_command("pip install -r requirements.txt --upgrade", "Update requirements")

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "install"
    
    if command == "install":
        install_dependencies()
    elif command == "install-dev":
        install_dev_dependencies()
    elif command == "update":
        update_dependencies()
    else:
        print("Usage: python uv_integration.py [install|install-dev|update]")
        sys.exit(1)