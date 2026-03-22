import argparse
import sys
from pathlib import Path

from yt_dlp import YoutubeDL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_links(links_file: Path) -> list[str]:
    if not links_file.exists():
        raise FileNotFoundError(f"Links file not found: {links_file}")

    links = []
    for raw_line in links_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        links.append(line)

    if not links:
        raise RuntimeError(f"No valid links found in {links_file}")

    return links


def build_downloader(output_dir: Path) -> YoutubeDL:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "download_archive.txt"

    options = {
        "outtmpl": str(output_dir / "%(title).140B [%(id)s].%(ext)s"),
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
        "restrictfilenames": True,
        "ignoreerrors": False,
        "quiet": False,
        "no_warnings": False,
        "progress_with_newline": True,
        "download_archive": str(archive_path),
    }

    return YoutubeDL(options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download gameplay videos with yt-dlp")
    parser.add_argument("--links_file", type=str, default="src/videos/links.txt", help="Text file containing one URL per line")
    parser.add_argument("--output_dir", type=str, default="src/videos", help="Directory where downloaded videos should be saved")
    args = parser.parse_args()

    links_file = Path(args.links_file)
    output_dir = Path(args.output_dir)

    links = load_links(links_file)
    downloader = build_downloader(output_dir)

    print(f"[Download] Loaded {len(links)} link(s) from {links_file}")
    print(f"[Download] Saving videos into {output_dir}")

    for index, link in enumerate(links, start=1):
        print(f"\n[Download] ({index}/{len(links)}) {link}")
        downloader.download([link])

    print("\n[Download] All videos completed successfully.")


if __name__ == "__main__":
    main()
