# 🏷️ Smart File Labeler

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An intelligent file renaming tool that automatically extracts names from metadata (EXIF, ID3, MP4 tags) and organizes your media collection. Perfect for photographers, musicians, and digital hoarders!

## ✨ Features

- **🔍 Metadata Extraction** - Reads EXIF (images), ID3 (audio), and MP4 tags (video)
- **🏷️ Intelligent Renaming** - Uses metadata when available, fallback to clean names
- **📁 Multi-folder Support** - Process multiple directories in one run
- **🛡️ Safe Operations** - Dry-run mode, collision detection, and detailed logging
- **📊 Comprehensive Reports** - JSON results with full operation details
- **🚀 Cross-Platform** - Works on Windows, macOS, and Linux
- **⚡ Fast & Efficient** - Processes thousands of files quickly

## 📁 Supported Formats

| Category | Formats | Metadata Sources |
|----------|---------|------------------|
| **Images** | JPG, PNG, HEIC, TIFF, WebP, GIF | EXIF, XMP, IPTC |
| **Audio** | MP3, M4A, FLAC, WAV, OGG, WMA | ID3, MP4 tags, Vorbis comments |
| **Video** | MP4, MOV, AVI, MKV, WMV | MP4 metadata, ASF tags |

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   
   git clone https://github.com/yourusername/smart-file-labeler.git
   cd smart-file-labeler


2. Install dependencies:

pip install -r requirements.txt


Basic Usage

Rename files in default folders (music, videos, images):

python file_labeler.py --base-folder /path/to/your/media


Dry-run (simulation only):

python file_labeler.py --dry-run --verbose


Custom folders and prefix:

python file_labeler.py --folders photos music --prefix "Uncategorized"


📊 Example Transformation
Before:

vacation_2023/
├── DSC_001.jpg           (EXIF: "Beach Sunset")
├── song123.mp3           (ID3: "Blinding Lights")
└── video001.mp4          (MP4: "Birthday Party")

After:

vacation_2023/
├── Beach_Sunset.jpg
├── Blinding_Lights.mp3
└── Birthday_Party.mp4


🛠️ Advanced Usage
Custom Configuration
Create a configuration file for repeated use:

# Create alias for common operations
alias organize-media="python file_labeler.py --base-folder ~/Media --folders photos music videos --prefix 'Unsorted'"


Integration with Other Tools
Use as a Python module:

from file_labeler import FileLabeler, MetadataExtractor

# Extract metadata without renaming
extractor = MetadataExtractor()
metadata = extractor.extract_metadata_name(Path("your_file.jpg"))

# Batch processing
labeler = FileLabeler("/media", ["photos", "music"])
results = labeler.process_all_folders(dry_run=True)


Shell Script Integration
Create a cleanup script:

# organize_media.sh
BASE_DIR="$1"
python file_labeler.py --base-folder "$BASE_DIR" --folders images audio video --force


📁 Project Structure

smart-file-labeler/
├── file_labeler.py          # Main script
├── requirements.txt         # Dependencies
├── README.md               # This file
├── examples/               # Usage examples
│   ├── before/             # Sample files before
│   └── after/              # Sample files after
├── tests/                  # Test suite
│   ├── test_metadata.py    # Metadata tests
│   └── test_renamer.py     # Renaming tests
└── results/                # Generated reports


⚙️ Configuration Options
Option	Description	Default

--base-folder	Root directory for media	Current directory
--folders	Subdirectories to process	music videos images
--prefix	Name for files without metadata	Unlabeled
--dry-run	Simulate without changes	False
--verbose	Detailed logging	False
--force	Skip confirmation	False
--output-format	Results format	json


🤝 Contributing
We welcome contributions! Here's how to help:

Report Issues - Found a bug? Open an issue with details
Request Features - Have an idea? Share it!
Submit Code - Fix bugs or add features via pull requests


Development Setup

# Clone and setup development environment
git clone https://github.com/yourusername/smart-file-labeler.git
cd smart-file-labeler

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black file_labeler.py


Adding Support for New Formats

Add extension to appropriate set in MetadataExtractor
Implement extraction method for the format
Add tests in tests/test_metadata.py


🙏 Acknowledgments
Pillow for image processing
mutagen for audio/video metadata
All contributors and users who help improve this tool


⚠️ Disclaimer
Always backup your files before running bulk renaming operations. The authors are not responsible for any data loss.


Made with ❤️ for organized digital lives
⭐ If this tool helps you, please consider starring the repository!
