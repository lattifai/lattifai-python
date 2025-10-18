#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-install the latest k2 wheel that matches the current machine.
- Prints in English.
- Sources:
  Linux CUDA wheels:     https://k2-fsa.github.io/k2/installation/pre-compiled-cuda-wheels-linux/index.html
  macOS CPU wheels:      https://k2-fsa.github.io/k2/installation/pre-compiled-cpu-wheels-macos/index.html
  Windows CPU wheels:    https://k2-fsa.github.io/k2/installation/pre-compiled-cpu-wheels-windows/index.html

Usage:
  python install_k2_auto.py            # install immediately
  python install_k2_auto.py --dry-run  # only show what would be installed
"""

import argparse
import os
import platform
import re
import subprocess
import sys
import urllib.request
from html.parser import HTMLParser
from typing import List, Optional, Tuple

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

CUDA_LINUX_URL = 'https://k2-fsa.github.io/k2/installation/pre-compiled-cuda-wheels-linux/index.html'
MAC_CPU_URL = 'https://k2-fsa.github.io/k2/installation/pre-compiled-cpu-wheels-macos/index.html'
WIN_CPU_URL = 'https://k2-fsa.github.io/k2/installation/pre-compiled-cpu-wheels-windows/index.html'


class WheelLinkParser(HTMLParser):
    def __init__(self, parse_mode='wheels'):
        super().__init__()
        self.links: List[str] = []
        self.parse_mode = parse_mode  # 'wheels' or 'versions'

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            href = dict(attrs).get('href')
            if href:
                if self.parse_mode == 'wheels' and href.endswith('.whl'):
                    self.links.append(href)
                elif self.parse_mode == 'versions' and re.match(r'^\d+\.\d+\.\d+\.html$', href):
                    # Match version links like "2.8.0.html"
                    self.links.append(href)


def fetch_wheel_links(
    page_url: str, target_torch_version: Optional[str] = None, cuda_version: Optional[str] = None
) -> List[str]:
    """
    Fetch wheel links from k2 pages. The structure is:
    - Index page contains links to version-specific pages (e.g., 2.8.0.html)
    - Version pages contain actual .whl file links

    Args:
        page_url: The base URL to fetch wheels from
        target_torch_version: If specified, only fetch wheels for this torch version (e.g., "2.8.0")
        cuda_version: If specified, prefer wheels with this CUDA version (e.g., "12.1")
    """
    with urllib.request.urlopen(page_url) as resp:
        html = resp.read().decode('utf-8', errors='ignore')

    # First, try to find version page links
    version_parser = WheelLinkParser(parse_mode='versions')
    version_parser.feed(html)

    if version_parser.links:
        # If we found version links, this is an index page
        # Filter version links if target_torch_version is specified
        version_links_to_process = version_parser.links
        if target_torch_version:
            target_filename = f'{target_torch_version}.html'
            version_links_to_process = [link for link in version_parser.links if link == target_filename]
            if not version_links_to_process:
                print(f'[WARN] No page found for torch version {target_torch_version}')
                return []
            print(f'[INFO] Found torch version {target_torch_version}, fetching wheels from {target_filename}')
        else:
            # If no target version specified, choose the highest version (latest torch version)
            def parse_version_from_link(link: str) -> Tuple[int, int, int]:
                # Extract version from "2.8.0.html" -> (2, 8, 0)
                match = re.match(r'^(\d+)\.(\d+)\.(\d+)\.html$', link)
                if match:
                    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                return (0, 0, 0)

            # Sort by version and take the highest
            sorted_versions = sorted(version_parser.links, key=parse_version_from_link, reverse=True)
            if sorted_versions:
                latest_version = sorted_versions[0]
                version_links_to_process = [latest_version]
                version_str = latest_version.replace('.html', '')
                print(f'[INFO] No target torch version specified, using latest version: {version_str}')

        # Fetch wheel links from version pages
        all_wheel_links = []
        base_url = page_url.rsplit('/', 1)[0]
        py_tag, abi_tag = py_tags()

        for version_link in version_links_to_process:
            version_url = f'{base_url}/{version_link}'
            try:
                with urllib.request.urlopen(version_url) as resp:
                    version_html = resp.read().decode('utf-8', errors='ignore')
                wheel_parser = WheelLinkParser(parse_mode='wheels')
                wheel_parser.feed(version_html)

                # If target version specified or using latest version, find matching wheels
                if target_torch_version or len(version_links_to_process) == 1:
                    matching_wheels = []
                    for wheel_link in wheel_parser.links:
                        if py_tag in wheel_link and abi_tag in wheel_link:
                            matching_wheels.append(wheel_link)

                    if cuda_version and matching_wheels:
                        # First try to find wheels with the specified CUDA version
                        cuda_specific_wheels = []
                        for wheel_link in matching_wheels:
                            wheel_cuda = parse_cuda_from_filename(wheel_link)
                            if wheel_cuda and wheel_cuda == cuda_version:
                                cuda_specific_wheels.append(wheel_link)

                        if cuda_specific_wheels:
                            # Found wheels with specified CUDA version, pick the latest one by dev date
                            def sort_by_devdate(wheel: str) -> int:
                                return parse_devdate(wheel) or 0

                            best_wheel = max(cuda_specific_wheels, key=sort_by_devdate)
                            print(
                                f'[INFO] Found matching wheel for Python {py_tag} and CUDA {cuda_version}: {best_wheel}'
                            )
                            return [best_wheel] if best_wheel.startswith('http') else [best_wheel]
                        else:
                            print(f'[WARN] No wheel found for CUDA {cuda_version}, falling back to latest version')

                    # If no CUDA version specified or no matching CUDA wheels found, use the latest wheel
                    if matching_wheels:

                        def sort_by_devdate(wheel: str) -> int:
                            return parse_devdate(wheel) or 0

                        best_wheel = max(matching_wheels, key=sort_by_devdate)
                        cuda_info = (
                            f' (CUDA {parse_cuda_from_filename(best_wheel)})'
                            if parse_cuda_from_filename(best_wheel)
                            else ''
                        )
                        print(f'[INFO] Found matching wheel for Python {py_tag}{cuda_info}: {best_wheel}')
                        return [best_wheel]

                    version_str = version_link.replace('.html', '')
                    print(f'[WARN] No wheel found for Python {py_tag} in torch {version_str}')
                else:
                    all_wheel_links.extend(wheel_parser.links)
            except Exception as e:
                print(f'[WARN] Failed to fetch {version_url}: {e}')
                continue

        # If target version specified or latest version but no matching wheel found
        if target_torch_version or len(version_links_to_process) == 1:
            return []

        # Normalize to absolute URLs for all wheels case
        abs_links = []
        for href in all_wheel_links:
            if href.startswith('http://') or href.startswith('https://'):
                abs_links.append(href)
            else:
                # For huggingface links, they are already absolute in the href
                abs_links.append(href)
        return abs_links

    else:
        raise ValueError('No version links found on the page; unexpected page structure.')


def py_tags() -> Tuple[str, str]:
    """Return (py_tag, abi_tag), e.g. ('cp310', 'cp310') for CPython."""
    impl = platform.python_implementation().lower()
    if impl != 'cpython':
        # Wheels are for CPython; still try cpXY
        pass
    major, minor = sys.version_info.major, sys.version_info.minor
    tag = f'cp{major}{minor}'
    return tag, tag


def detect_torch_version() -> Optional[str]:
    """
    Detect installed PyTorch version string like '2.8.0'.
    Returns None if PyTorch is not installed.
    """
    try:
        import importlib

        torch = importlib.import_module('torch')
        version = getattr(torch, '__version__', None)
        if version:
            # Extract major.minor.patch from version string (remove +cu118 etc suffixes)
            version_match = re.match(r'(\d+\.\d+\.\d+)', str(version))
            if version_match:
                return version_match.group(1)
    except Exception:
        pass
    return None


def detect_cuda_version_linux() -> Optional[str]:
    """
    Detect CUDA version string like '12.1'.
    Priority: torch.version.cuda -> nvidia-smi -> None
    """
    # Try PyTorch if installed
    try:
        import importlib

        torch = importlib.import_module('torch')
        v = getattr(getattr(torch, 'version', None), 'cuda', None)
        if v:
            return str(v)
    except Exception:
        pass

    # # Try nvidia-smi
    # try:
    #     out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
    #     m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    #     if m:
    #         return m.group(1)
    # except Exception:
    #     pass

    return None


def parse_cuda_from_filename(name: str) -> Optional[str]:
    # e.g., ...+cuda12.1-..., ...+cuda11.8-...
    m = re.search(r'cuda(\d+(?:\.\d+)?)', name)
    return m.group(1) if m else None


def parse_devdate(name: str) -> Optional[int]:
    # e.g., dev20240606
    m = re.search(r'dev(\d{8})', name)
    return int(m.group(1)) if m else None


def parse_version_tuple(name: str) -> Tuple[int, ...]:
    # k2-<version>...   take first contiguous version-like sequence
    m = re.search(r'k2-([\d]+(?:\.[\d]+)*)', name)
    if not m:
        return tuple()
    return tuple(int(p) for p in m.group(1).split('.'))


def best_match_cuda(candidates: List[str], installed_cuda: Optional[str]) -> List[str]:
    """
    Keep only CUDA wheels; if installed_cuda present, prefer same major.minor,
    else fallback to highest CUDA version available.
    """
    cuda_wheels = [w for w in candidates if 'cuda' in w.lower()]
    if not cuda_wheels:
        return []

    if installed_cuda:
        # Normalize like '12.1' -> (12,1)
        def to_tuple(v: str) -> Tuple[int, int]:
            parts = v.split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            return (major, minor)

        target = to_tuple(installed_cuda)

        # Score by distance in (major, minor); prefer exact or closest lower/higher
        def score(w: str) -> Tuple[int, int, int]:
            wc = parse_cuda_from_filename(w) or '0'
            wt = to_tuple(wc)
            # absolute distance
            dist = (abs(wt[0] - target[0]) * 100) + abs(wt[1] - target[1])
            # Prefer same major, then higher minor not exceeding target, etc.
            bias = 0 if wt[0] == target[0] else 1
            # Negative if <= target to prefer not exceeding
            not_exceed = 0 if (wt <= target) else 1
            return (dist, bias, not_exceed)

        cuda_wheels.sort(key=score)
        # Keep top-N that share the best CUDA version string (for later date/version sorting)
        best_cuda = parse_cuda_from_filename(cuda_wheels[0])
        cuda_wheels = [w for w in cuda_wheels if parse_cuda_from_filename(w) == best_cuda]
        return cuda_wheels

    # No installed CUDA detected: pick the highest CUDA in page (by version tuple)
    def cudatuple(w: str) -> Tuple[int, int]:
        c = parse_cuda_from_filename(w) or '0'
        parts = c.split('.')
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)

    cuda_wheels.sort(key=cudatuple, reverse=True)
    top = parse_cuda_from_filename(cuda_wheels[0])
    return [w for w in cuda_wheels if parse_cuda_from_filename(w) == top]


def platform_tag_filters() -> List[str]:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == 'linux':
        # Manylinux tags typically include 'linux_x86_64' or 'manylinux...' but
        # the page often lists 'linux_x86_64'. We'll match the common substrings.
        if 'aarch64' in machine or 'arm64' in machine:
            return ['linux_aarch64', 'manylinux_aarch64']
        return ['linux_x86_64', 'manylinux_x86_64']

    if system == 'darwin':
        if 'arm64' in machine or 'aarch64' in machine:
            return ['macosx_11_0_arm64', 'macosx_12_0_arm64', 'macosx_13_0_arm64', 'macosx_14_0_arm64']
        # Intel macs
        return [
            'macosx_10_9_x86_64',
            'macosx_11_0_x86_64',
            'macosx_12_0_x86_64',
            'macosx_13_0_x86_64',
            'macosx_14_0_x86_64',
        ]

    if system == 'windows':
        if 'arm64' in machine:
            # If k2 provides win_arm64 wheels in future, this will catch them.
            return ['win_arm64']
        return ['win_amd64']

    return []


def choose_best_wheel(links: List[str], require_cuda: bool) -> Optional[str]:
    py_tag, abi_tag = py_tags()
    plat_filters = platform_tag_filters()

    def match_basic(name: str) -> bool:
        # python tag & abi tag must appear
        if py_tag not in name or abi_tag not in name:
            return False
        # platform tag must match one of known substrings
        if not any(tag in name for tag in plat_filters):
            return False
        return True

    candidates = [u for u in links if match_basic(u)]
    if not candidates:
        return None

    if require_cuda:
        candidates = best_match_cuda(candidates, detect_cuda_version_linux())
        if not candidates:
            return None
    else:
        # For CPU, try to exclude CUDA wheels explicitly
        candidates = [u for u in candidates if 'cuda' not in u.lower()]

    # Now sort by (dev date desc, version desc, URL lex desc as tie-breaker)
    def sort_key(u: str):
        date = parse_devdate(u) or 0
        ver = parse_version_tuple(u)
        return (date, ver, u)

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0] if candidates else None


def run_pip_install(wheel_url: str, dry_run: bool):
    cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', '--no-cache-dir', wheel_url]
    print('[INFO] Pip command:', ' '.join(cmd))
    if dry_run:
        print('[DRY-RUN] Skipping actual installation.')
        return
    try:
        subprocess.check_call(cmd)
        print('[SUCCESS] k2 has been installed successfully.')
    except subprocess.CalledProcessError as e:
        print('[ERROR] pip install failed with exit code:', e.returncode)
        sys.exit(e.returncode)


def install_k2_main(dry_run: bool = False):
    """Main function to install k2 without argparse, suitable for programmatic use."""
    system = platform.system().lower()
    print(f'[INFO] Detected OS: {system}')
    print(f'[INFO] Python: {platform.python_version()} | Impl: {platform.python_implementation()}')

    # Check if torch is already installed
    torch_version = detect_torch_version()
    if torch_version:
        print(f'[INFO] Detected PyTorch version: {torch_version}')
    else:
        print('[INFO] PyTorch not detected, will search all available versions')

    if system == 'linux':
        print('[INFO] Target: Linux (CUDA wheels)')
        cuda_version = detect_cuda_version_linux()
        if not cuda_version:
            print('[WARN] No CUDA detected on Linux.')
            # print("[HINT] Install CUDA or build from source if CPU-only is required.")
            # print("")
            # print("To build k2 from source, you can run the following commands:")
            # print("  git clone https://github.com/k2-fsa/k2.git")
            # print("  cd k2")
            # print('  export K2_MAKE_ARGS="-j6"')
            # print("  python3 setup.py install")
            # print("")
            # response = input("Do you want to continue with source installation? (y/N): ").strip().lower()
            # if response in ["y", "yes"]:
            #     print("[INFO] Please run the commands above manually to install k2 from source.")
            # sys.exit(2)
        print(f'[INFO] Detected CUDA version: {cuda_version}')

        wheel = None
        for _torch_version in [torch_version, None] if torch_version else [None]:
            for _cuda_version in [cuda_version, None] if cuda_version else [None]:
                links = fetch_wheel_links(CUDA_LINUX_URL, _torch_version, cuda_version=_cuda_version)
                if _torch_version and links:
                    # If we have torch version and found matching wheel, use it directly
                    wheel = links[0]
                else:
                    # Fallback to traditional selection
                    if not links:
                        links = fetch_wheel_links(CUDA_LINUX_URL)
                    wheel = choose_best_wheel(links, require_cuda=_cuda_version is not None)

                    if not _torch_version and links and not wheel:
                        wheel = links[0]  # Pick first available as last resort

                if not wheel:
                    if _cuda_version:
                        print(
                            f'[WARN] No suitable wheel found for CUDA {_cuda_version}, " + \
                            "trying without CUDA preference...'
                        )
                else:
                    break  # Found a wheel, exit loop

            if not wheel and _torch_version:
                print(
                    f'[WARN] Tried torch version {_torch_version}, but not found wheel, trying without torch version...'
                )

            if wheel:
                break

        print(f'[INFO] Selected wheel:\n  {wheel}')
        run_pip_install(wheel, dry_run)
        return

    elif system == 'darwin':
        print('[INFO] Target: macOS (CPU wheels)')
        for _torch_version in [torch_version, None] if torch_version else [None]:
            links = fetch_wheel_links(MAC_CPU_URL, _torch_version)
            if _torch_version and links:
                # If we have torch version and found matching wheel, use it directly
                wheel = links[0]
            else:
                # Fallback to traditional selection
                if not links:
                    links = fetch_wheel_links(MAC_CPU_URL)
                wheel = choose_best_wheel(links, require_cuda=False)
                if links and not wheel:
                    wheel = links[0]  # Pick first available as last resort

        if not wheel:
            print('[ERROR] Could not find a suitable macOS CPU wheel for your Python/platform.')
            sys.exit(1)

        print(f'[INFO] Selected wheel:\n  {wheel}')
        run_pip_install(wheel, dry_run)
        return

    elif system == 'windows':
        print('[INFO] Target: Windows (CPU wheels)')
        for _torch_version in [torch_version, None] if torch_version else [None]:
            links = fetch_wheel_links(WIN_CPU_URL, torch_version)
            if torch_version and links:
                # If we have torch version and found matching wheel, use it directly
                wheel = links[0]
            else:
                # Fallback to traditional selection
                if not links:
                    links = fetch_wheel_links(WIN_CPU_URL)
                wheel = choose_best_wheel(links, require_cuda=False)
                if links and not wheel:
                    wheel = links[0]  # Pick first available as last resort

        if not wheel:
            print('[ERROR] Could not find a suitable Windows CPU wheel for your Python/platform.')
            sys.exit(1)
        print(f'[INFO] Selected wheel:\n  {wheel}')
        run_pip_install(wheel, dry_run)
        return

    else:
        print(f'[ERROR] Unsupported OS: {system}')
        sys.exit(3)


def install_k2():
    """CLI entry point with argparse support."""
    parser = argparse.ArgumentParser(description='Auto-install the latest k2 wheel for your environment.')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be installed without making changes.')
    args = parser.parse_args()
    install_k2_main(dry_run=args.dry_run)


if __name__ == '__main__':
    install_k2()
