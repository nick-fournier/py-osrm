#!/usr/bin/env python3
"""
Auto-installer for py-osrm from GitHub Releases
Detects platform and Python version, then installs appropriate wheel
Falls back to git installation if no matching wheel is found
"""

import sys
import platform
import subprocess
import json
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

REPO_OWNER = "nick-fournier"
REPO_NAME = "py-osrm"
BRANCH = "revival"

def get_platform_tag():
    """Detect platform and architecture tag for wheel filename"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Map to wheel platform tags
    if system == "linux":
        if machine in ["x86_64", "amd64"]:
            return "linux_x86_64"
        elif machine in ["aarch64", "arm64"]:
            return "linux_aarch64"
    elif system == "darwin":
        if machine in ["x86_64", "amd64"]:
            return "macosx_10_9_x86_64"
        elif machine in ["arm64"]:
            return "macosx_11_0_arm64"
    elif system == "windows":
        if machine in ["x86_64", "amd64"]:
            return "win_amd64"
    
    return None

def get_python_tag():
    """Get Python version tag for wheel filename"""
    version_info = sys.version_info
    # py-osrm uses abi3 for stable ABI (Python 3.9+)
    if version_info >= (3, 9):
        return "cp39-abi3"
    return None

def get_latest_release():
    """Fetch latest release info from GitHub API"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
    try:
        req = Request(url, headers={"Accept": "application/vnd.github.v3+json"})
        with urlopen(req, timeout=10) as response:
            return json.loads(response.read())
    except (HTTPError, URLError) as e:
        print(f"⚠️  Could not fetch releases from GitHub: {e}")
        return None

def find_matching_wheel(release_data):
    """Find wheel matching current platform and Python version"""
    if not release_data or "assets" not in release_data:
        return None
    
    platform_tag = get_platform_tag()
    python_tag = get_python_tag()
    
    if not platform_tag or not python_tag:
        print(f"⚠️  Platform not detected or Python version < 3.9")
        return None
    
    # Look for matching wheel
    for asset in release_data["assets"]:
        name = asset["name"]
        if name.endswith(".whl") and python_tag in name and platform_tag in name:
            return asset["browser_download_url"]
    
    return None

def install_wheel(wheel_url):
    """Install wheel using pip"""
    print(f"📦 Installing from wheel: {wheel_url}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_url])
        print("✅ Installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def install_from_git():
    """Fallback: Install from git (requires compilation)"""
    git_url = f"git+https://github.com/{REPO_OWNER}/{REPO_NAME}.git@{BRANCH}"
    print(f"\n⚠️  No pre-built wheel found for your platform.")
    print(f"📦 Falling back to git installation (this will compile from source)")
    print(f"⏱️  This may take 5-10 minutes and requires C++ compiler and CMake...")
    print(f"\nInstalling from: {git_url}\n")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", git_url])
        print("\n✅ Installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        print("\nPlease ensure you have:")
        print("  - C++ compiler (gcc/g++ on Linux, Xcode tools on macOS, MSVC on Windows)")
        print("  - CMake (install via: pip install cmake)")
        return False

def main():
    print(f"🔍 Detecting platform: {platform.system()} {platform.machine()}")
    print(f"🐍 Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Try to get latest release
    print(f"\n🔄 Fetching latest release from GitHub...")
    release_data = get_latest_release()
    
    if release_data:
        version = release_data.get("tag_name", "unknown")
        print(f"📌 Latest release: {version}")
        
        # Try to find matching wheel
        wheel_url = find_matching_wheel(release_data)
        if wheel_url:
            if install_wheel(wheel_url):
                return 0
    
    # Fallback to git installation
    if install_from_git():
        return 0
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
