#!/usr/bin/env python3
"""
Download YOLOv3 weights and config files
"""

import urllib.request
import sys
from pathlib import Path

def download_file(url, destination, description):
    """Download file with progress bar"""
    print(f"\n📥 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Saving to: {destination}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        size_mb = total_size / (1024 * 1024)
        downloaded_mb = (count * block_size) / (1024 * 1024)

        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)

        sys.stdout.write(f'\r   [{bar}] {percent}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)')
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print(f"\n   ✅ Downloaded successfully!\n")
        return True
    except Exception as e:
        print(f"\n   ❌ Error: {e}\n")
        return False

def main():
    # Create model_data directory
    model_dir = Path("model_data")
    model_dir.mkdir(exist_ok=True)

    print("="*60)
    print("🚗 YOLO Model Download")
    print("="*60)

    # Files to download
    files = [
        {
            "url": "https://pjreddie.com/media/files/yolov3.weights",
            "dest": model_dir / "yolov3.weights",
            "desc": "YOLOv3 Weights (248 MB)"
        },
        {
            "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            "dest": model_dir / "yolov3.cfg",
            "desc": "YOLOv3 Config"
        },
        {
            "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            "dest": model_dir / "coco_classes.txt",
            "desc": "COCO Class Names"
        }
    ]

    # Download each file
    success_count = 0
    for file_info in files:
        if file_info["dest"].exists():
            size_mb = file_info["dest"].stat().st_size / (1024 * 1024)
            if size_mb > 1:  # If file is larger than 1MB, assume it's complete
                print(f"\n✓ {file_info['desc']} already exists ({size_mb:.1f} MB)")
                success_count += 1
                continue

        if download_file(file_info["url"], file_info["dest"], file_info["desc"]):
            success_count += 1

    # Summary
    print("="*60)
    if success_count == len(files):
        print("✅ ALL FILES DOWNLOADED SUCCESSFULLY!")
        print("="*60)
        print("\nYou can now run:")
        print("   python3 detect_real.py --image images/example.jpg")
        print("\nThis will use REAL YOLO detection!")
    else:
        print(f"⚠️  Downloaded {success_count}/{len(files)} files")
        print("="*60)

if __name__ == "__main__":
    main()
