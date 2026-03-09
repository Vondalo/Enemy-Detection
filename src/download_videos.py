"""
Download YouTube videos listed in a text file into the /videos folder.

Usage:
    python src/download_videos.py                         # uses default src/videos/links.txt
    python src/download_videos.py path/to/my_links.txt    # custom links file
"""

import os
import sys
import subprocess

# -------------------------
# CONFIGURATION
# -------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LINKS_FILE = os.path.join(SCRIPT_DIR, "videos", "links.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "videos")


def ensure_yt_dlp():
    """Make sure yt-dlp is installed."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("yt-dlp not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=True)


def read_links(filepath):
    """Read non-empty, non-comment lines from a text file."""
    if not os.path.exists(filepath):
        print(f"Error: Links file not found: {filepath}")
        sys.exit(1)

    links = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                links.append(line)
    return links


def download_video(url, output_dir):
    """Download a single video using yt-dlp."""
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", os.path.join(output_dir, "%(title)s.%(ext)s"),
        "--no-overwrites",           # skip if file already exists
        "--restrict-filenames",      # safe filenames (no spaces/special chars)
        url,
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    links_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LINKS_FILE

    print(f"Reading links from: {links_file}")
    links = read_links(links_file)

    if not links:
        print("No links found in file. Add YouTube URLs (one per line) and try again.")
        return

    print(f"Found {len(links)} link(s). Downloading to: {OUTPUT_DIR}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ensure_yt_dlp()

    success = 0
    failed = []
    for i, url in enumerate(links, 1):
        print(f"\n[{i}/{len(links)}] Downloading: {url}")
        if download_video(url, OUTPUT_DIR):
            success += 1
        else:
            failed.append(url)

    print(f"\n{'='*50}")
    print(f"Done! {success}/{len(links)} videos downloaded successfully.")
    if failed:
        print("Failed URLs:")
        for url in failed:
            print(f"  - {url}")


if __name__ == "__main__":
    main()
