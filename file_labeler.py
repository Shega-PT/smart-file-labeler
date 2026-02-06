#!/usr/bin/env python3
"""
Smart File Labeler - Intelligent file renaming tool based on metadata extraction.
Automatically renames media files using embedded metadata or generates clean names.
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Third-party imports
from PIL import Image, UnidentifiedImageError
import mutagen
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.asf import ASF
from mutagen.id3 import ID3

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure logging for the file labeler.
    
    Args:
        verbose: Enable debug logging if True
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('file_labeler')
    
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = Path('file_labeler.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setFormatter(console_format)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RenameOperation:
    """Represents a single file rename operation."""
    old_path: Path
    new_path: Path
    metadata_name: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class ProcessingResults:
    """Aggregates results from file processing."""
    renamed: List[RenameOperation] = None
    skipped: List[Path] = None
    errors: List[str] = None
    no_metadata_count: int = 0
    name_collisions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.renamed is None:
            self.renamed = []
        if self.skipped is None:
            self.skipped = []
        if self.errors is None:
            self.errors = []
        if self.name_collisions is None:
            self.name_collisions = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            'renamed': [asdict(op) for op in self.renamed],
            'skipped': [str(p) for p in self.skipped],
            'errors': self.errors,
            'no_metadata_count': self.no_metadata_count,
            'name_collisions': self.name_collisions,
            'summary': self.summary()
        }
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            'total_processed': len(self.renamed) + len(self.skipped) + len(self.errors),
            'successfully_renamed': len([op for op in self.renamed if op.success]),
            'skipped_files': len(self.skipped),
            'failed_operations': len(self.errors),
            'files_without_metadata': self.no_metadata_count,
            'name_collisions_detected': len(self.name_collisions)
        }


# ============================================================================
# METADATA EXTRACTOR
# ============================================================================

class MetadataExtractor:
    """
    Extracts metadata from various file types for intelligent renaming.
    Supports images, audio, and video files.
    """
    
    # Supported file formats
    AUDIO_EXTENSIONS = {'.mp3', '.m4a', '.m4p', '.wav', '.flac', '.aac', '.ogg', '.wma'}
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic'}
    
    UNWANTED_NAMES = {'', 'unknown', 'untitled', 'none', 'null', 'undefined'}
    
    @staticmethod
    def sanitize_filename(name: str) -> str:
        """
        Clean and normalize a filename to be filesystem-safe.
        
        Args:
            name: Original filename
            
        Returns:
            Sanitized filename
        """
        if not name:
            return ""
        
        # Normalize Unicode characters
        name = name.strip()
        
        # Replace problematic characters
        invalid_chars = r'<>:"/\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        
        # Replace control characters
        name = ''.join(char for char in name if ord(char) >= 32)
        
        # Collapse multiple spaces/underscores
        name = re.sub(r'[\s_]+', '_', name)
        
        # Remove trailing dots and spaces
        name = name.rstrip('. ')
        
        # Ensure reasonable length
        if len(name) > 200:
            name = name[:200]
        
        return name
    
    @classmethod
    def get_file_category(cls, file_path: Path) -> str:
        """
        Determine file category based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File category: 'audio', 'video', 'image', or 'other'
        """
        extension = file_path.suffix.lower()
        
        if extension in cls.AUDIO_EXTENSIONS:
            return 'audio'
        elif extension in cls.VIDEO_EXTENSIONS:
            return 'video'
        elif extension in cls.IMAGE_EXTENSIONS:
            return 'image'
        else:
            return 'other'
    
    @classmethod
    def extract_metadata_name(cls, file_path: Path) -> Optional[str]:
        """
        Extract a name from file metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted name or None if not found
        """
        category = cls.get_file_category(file_path)
        
        try:
            if category == 'audio':
                return cls._extract_audio_name(file_path)
            elif category == 'video':
                return cls._extract_video_name(file_path)
            elif category == 'image':
                return cls._extract_image_name(file_path)
        except Exception as e:
            logging.debug(f"Failed to extract metadata from {file_path}: {e}")
        
        return None
    
    @staticmethod
    def _extract_audio_name(file_path: Path) -> Optional[str]:
        """
        Extract name from audio file metadata.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Extracted name or None
        """
        extension = file_path.suffix.lower()
        
        try:
            # MP3 files with ID3 tags
            if extension == '.mp3':
                try:
                    audio = ID3(file_path)
                    if 'TIT2' in audio:  # Title tag
                        title = str(audio['TIT2']).strip()
                        if title and title.lower() not in MetadataExtractor.UNWANTED_NAMES:
                            return title
                except:
                    # Fallback to mutagen
                    audio = MP3(file_path)
                    if audio.tags:
                        for tag in ['TIT2', 'title', 'Title']:
                            if tag in audio.tags:
                                title = str(audio.tags[tag]).strip()
                                if title and title.lower() not in MetadataExtractor.UNWANTED_NAMES:
                                    return title
            
            # MP4/M4A files
            elif extension in ['.m4a', '.m4p', '.mp4']:
                audio = MP4(file_path)
                if audio.tags:
                    for tag in ['©nam', 'title']:
                        if tag in audio.tags:
                            title = audio.tags[tag]
                            if isinstance(title, list):
                                title = title[0]
                            title = str(title).strip()
                            if title and title.lower() not in MetadataExtractor.UNWANTED_NAMES:
                                return title
            
            # WMA files
            elif extension == '.wma':
                audio = ASF(file_path)
                if hasattr(audio, 'tags') and 'Title' in audio.tags:
                    title = str(audio.tags['Title'][0]).strip()
                    if title and title.lower() not in MetadataExtractor.UNWANTED_NAMES:
                        return title
            
            # Other audio formats
            else:
                audio = mutagen.File(file_path, easy=True)
                if audio and hasattr(audio, 'tags'):
                    tags = audio.tags
                    for tag_key in ['title', 'Title', 'TITLE']:
                        if tag_key in tags:
                            title = tags[tag_key]
                            if isinstance(title, list):
                                title = title[0]
                            title = str(title).strip()
                            if title and title.lower() not in MetadataExtractor.UNWANTED_NAMES:
                                return title
        
        except Exception as e:
            logging.debug(f"Audio metadata extraction failed for {file_path}: {e}")
        
        return None
    
    @staticmethod
    def _extract_video_name(file_path: Path) -> Optional[str]:
        """
        Extract name from video file metadata.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Extracted name or None
        """
        extension = file_path.suffix.lower()
        
        try:
            # MP4/MOV files
            if extension in ['.mp4', '.mov', '.m4v']:
                video = MP4(file_path)
                if video.tags:
                    for tag in ['©nam', 'title']:
                        if tag in video.tags:
                            title = video.tags[tag]
                            if isinstance(title, list):
                                title = title[0]
                            title = str(title).strip()
                            if title and title.lower() not in MetadataExtractor.UNWANTED_NAMES:
                                return title
            
            # ASF/WMV files
            elif extension in ['.wmv', '.asf']:
                video = ASF(file_path)
                if hasattr(video, 'tags') and 'Title' in video.tags:
                    title = str(video.tags['Title'][0]).strip()
                    if title and title.lower() not in MetadataExtractor.UNWANTED_NAMES:
                        return title
            
            # Other video formats
            else:
                video = mutagen.File(file_path)
                if video and hasattr(video, 'tags'):
                    tags = video.tags
                    for tag_key in ['title', 'Title']:
                        if tag_key in tags:
                            title = tags[tag_key]
                            if isinstance(title, list):
                                title = title[0]
                            title = str(title).strip()
                            if title and title.lower() not in MetadataExtractor.UNWANTED_NAMES:
                                return title
        
        except Exception as e:
            logging.debug(f"Video metadata extraction failed for {file_path}: {e}")
        
        return None
    
    @staticmethod
    def _extract_image_name(file_path: Path) -> Optional[str]:
        """
        Extract name from image file metadata.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Extracted name or None
        """
        try:
            with Image.open(file_path) as img:
                # Try EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    # EXIF tags: 270=ImageDescription, 315=Artist, 305=Software
                    for tag in [270, 315, 305]:
                        if tag in exif:
                            value = str(exif[tag]).strip()
                            if value and value.lower() not in MetadataExtractor.UNWANTED_NAMES:
                                return value
                
                # Try XMP data (if available)
                if hasattr(img, 'info') and 'xmp' in img.info:
                    # Simple XMP extraction - look for title/description
                    xmp_data = img.info['xmp']
                    if isinstance(xmp_data, bytes):
                        xmp_data = xmp_data.decode('utf-8', errors='ignore')
                    
                    # Look for common XMP title patterns
                    patterns = [
                        r'<dc:title>.*?<rdf:li.*?>(.*?)</rdf:li>',
                        r'<dc:description>.*?<rdf:li.*?>(.*?)</rdf:li>',
                        r'photoshop:Headline="(.*?)"'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, xmp_data, re.IGNORECASE | re.DOTALL)
                        if match:
                            title = match.group(1).strip()
                            if title and title.lower() not in MetadataExtractor.UNWANTED_NAMES:
                                return title
        
        except Exception as e:
            logging.debug(f"Image metadata extraction failed for {file_path}: {e}")
        
        return None


# ============================================================================
# FILE LABELER
# ============================================================================

class FileLabeler:
    """
    Main class for intelligent file renaming based on metadata.
    """
    
    def __init__(self, base_folder: str, folders: Optional[List[str]] = None, prefix: str = "Unlabeled"):
        """
        Initialize the file labeler.
        
        Args:
            base_folder: Root directory containing media folders
            folders: List of folder names to process (None for default)
            prefix: Prefix for files without metadata
        """
        self.base_folder = Path(base_folder).resolve()
        self.prefix = prefix
        self.extractor = MetadataExtractor()
        
        # Default folders if not specified
        if folders is None:
            self.folders = ['music', 'videos', 'images', 'documents']
        else:
            self.folders = folders
        
        # Validate folders exist
        self.valid_folders = []
        for folder in self.folders:
            folder_path = self.base_folder / folder
            if folder_path.exists() and folder_path.is_dir():
                self.valid_folders.append(folder)
            else:
                logging.warning(f"Folder '{folder}' not found in {self.base_folder}")
    
    def generate_unique_filename(self, folder: Path, base_name: str, extension: str) -> str:
        """
        Generate a unique filename in the target folder.
        
        Args:
            folder: Target directory
            base_name: Base name without counter
            extension: File extension with dot
            
        Returns:
            Unique filename
        """
        counter = 0
        while True:
            if counter == 0:
                filename = f"{base_name}{extension}"
            else:
                filename = f"{base_name}_{counter}{extension}"
            
            if not (folder / filename).exists():
                return filename
            
            counter += 1
    
    def process_folder(self, folder_name: str, dry_run: bool = False) -> ProcessingResults:
        """
        Process all files in a specific folder.
        
        Args:
            folder_name: Name of folder to process
            dry_run: If True, only simulate renaming
            
        Returns:
            Processing results
        """
        results = ProcessingResults()
        folder_path = self.base_folder / folder_name
        used_names = {}  # Track names to detect collisions
        
        logging.info(f"Processing folder: {folder_name}")
        
        # Get all regular files
        files = [f for f in folder_path.iterdir() if f.is_file()]
        
        if not files:
            logging.info(f"  No files found in {folder_name}")
            return results
        
        logging.info(f"  Found {len(files)} files")
        
        for idx, file_path in enumerate(files, 1):
            logging.info(f"  [{idx}/{len(files)}] Analyzing: {file_path.name}")
            
            try:
                # Extract metadata name
                metadata_name = self.extractor.extract_metadata_name(file_path)
                
                # Determine base name to use
                if metadata_name:
                    clean_name = self.extractor.sanitize_filename(metadata_name)
                    if clean_name:
                        base_name = clean_name
                        logging.info(f"    Using metadata name: '{clean_name}'")
                    else:
                        base_name = self.prefix
                        results.no_metadata_count += 1
                        logging.info(f"    Invalid metadata, using: {self.prefix}")
                else:
                    base_name = self.prefix
                    results.no_metadata_count += 1
                    logging.info(f"    No metadata found, using: {self.prefix}")
                
                # Generate unique filename
                extension = file_path.suffix.lower()
                new_filename = self.generate_unique_filename(folder_path, base_name, extension)
                
                # Check for name collisions
                if new_filename in used_names:
                    results.name_collisions.append({
                        'old_name': file_path.name,
                        'new_name': new_filename,
                        'folder': folder_name
                    })
                
                used_names[new_filename] = used_names.get(new_filename, 0) + 1
                
                # Skip if name hasn't changed
                if new_filename == file_path.name:
                    logging.info(f"    Name already correct: {file_path.name}")
                    results.skipped.append(file_path)
                    continue
                
                # Create rename operation
                new_path = folder_path / new_filename
                operation = RenameOperation(
                    old_path=file_path,
                    new_path=new_path,
                    metadata_name=metadata_name
                )
                
                # Execute or simulate rename
                if dry_run:
                    logging.info(f"    [DRY RUN] Would rename: '{file_path.name}' -> '{new_filename}'")
                    results.renamed.append(operation)
                else:
                    try:
                        file_path.rename(new_path)
                        logging.info(f"    ✓ Renamed: '{file_path.name}' -> '{new_filename}'")
                        results.renamed.append(operation)
                    except Exception as e:
                        error_msg = f"Failed to rename {file_path}: {e}"
                        operation.success = False
                        operation.error = error_msg
                        results.renamed.append(operation)
                        results.errors.append(error_msg)
                        logging.error(f"    {error_msg}")
            
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                results.errors.append(error_msg)
                logging.error(f"    {error_msg}")
        
        return results
    
    def process_all_folders(self, dry_run: bool = False) -> ProcessingResults:
        """
        Process all valid folders.
        
        Args:
            dry_run: If True, only simulate renaming
            
        Returns:
            Aggregated processing results
        """
        all_results = ProcessingResults()
        
        if not self.valid_folders:
            logging.error("No valid folders found to process!")
            return all_results
        
        for folder_name in self.valid_folders:
            folder_results = self.process_folder(folder_name, dry_run)
            
            # Merge results
            all_results.renamed.extend(folder_results.renamed)
            all_results.skipped.extend(folder_results.skipped)
            all_results.errors.extend(folder_results.errors)
            all_results.no_metadata_count += folder_results.no_metadata_count
            all_results.name_collisions.extend(folder_results.name_collisions)
        
        return all_results
    
    def save_results(self, results: ProcessingResults, dry_run: bool = False) -> Optional[Path]:
        """
        Save processing results to JSON file.
        
        Args:
            results: Processing results to save
            dry_run: Whether this was a dry run
            
        Returns:
            Path to saved file or None
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "dry_run" if dry_run else "executed"
        filename = f"labeling_results_{timestamp}_{mode}.json"
        results_file = self.base_folder / filename
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
            
            logging.info(f"\nResults saved to: {results_file}")
            return results_file
        
        except Exception as e:
            logging.error(f"Failed to save results: {e}")
            return None


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart File Labeler - Rename files based on metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --base-folder /media --folders music videos
  %(prog)s --dry-run --verbose
  %(prog)s --prefix "Uncategorized" --folders downloads
        """
    )
    
    parser.add_argument(
        "--base-folder",
        default=".",
        help="Base folder containing media directories (default: current)"
    )
    
    parser.add_argument(
        "--folders",
        nargs="+",
        default=['music', 'videos', 'images'],
        help="Folders to process (default: music videos images)"
    )
    
    parser.add_argument(
        "--prefix",
        default="Unlabeled",
        help="Prefix for files without metadata (default: Unlabeled)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate renaming without making changes"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    parser.add_argument(
        "--output-format",
        choices=['json', 'text'],
        default='json',
        help="Output format for results (default: json)"
    )
    
    return parser.parse_args()


def confirm_operation(total_files: int, dry_run: bool) -> bool:
    """
    Ask for user confirmation before proceeding.
    
    Args:
        total_files: Number of files to process
        dry_run: Whether this is a dry run
        
    Returns:
        True if confirmed, False otherwise
    """
    if dry_run:
        print(f"\n📋 DRY RUN MODE: Will analyze {total_files} files")
        print("   No files will be renamed.")
    else:
        print(f"\n⚠️  ATTENTION: Will rename {total_files} files")
        print("   Files will be renamed based on metadata.")
        print("   Files without metadata will use the specified prefix.")
    
    response = input("\nContinue? (y/N): ").strip().lower()
    return response in ['y', 'yes', 'sim']


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Check dependencies
    try:
        import PIL
        import mutagen
    except ImportError as e:
        logger.error(f"Required libraries not found: {e}")
        logger.error("Install with: pip install Pillow mutagen")
        return 1
    
    try:
        # Initialize labeler
        labeler = FileLabeler(args.base_folder, args.folders, args.prefix)
        
        if not labeler.valid_folders:
            logger.error("No valid folders found!")
            return 1
        
        # Count total files
        total_files = 0
        for folder_name in labeler.valid_folders:
            folder_path = labeler.base_folder / folder_name
            files = [f for f in folder_path.iterdir() if f.is_file()]
            total_files += len(files)
        
        if total_files == 0:
            logger.info("No files found to process.")
            return 0
        
        # Show summary
        logger.info(f"\n📁 Smart File Labeler")
        logger.info(f"   Base folder: {labeler.base_folder}")
        logger.info(f"   Folders to process: {', '.join(labeler.valid_folders)}")
        logger.info(f"   Default prefix: {args.prefix}")
        logger.info(f"   Total files found: {total_files}")
        
        # Ask for confirmation
        if not args.dry_run and not args.force:
            if not confirm_operation(total_files, args.dry_run):
                logger.info("Operation cancelled.")
                return 0
        
        # Process files
        logger.info("\n" + "="*60)
        logger.info("STARTING FILE PROCESSING")
        logger.info("="*60)
        
        results = labeler.process_all_folders(args.dry_run)
        
        # Save results
        results_file = labeler.save_results(results, args.dry_run)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        
        summary = results.summary()
        logger.info(f"✅ Successfully renamed: {summary['successfully_renamed']}")
        logger.info(f"↻ Skipped (already correct): {summary['skipped_files']}")
        logger.info(f"❓ Without metadata: {summary['files_without_metadata']}")
        logger.info(f"⚠️  Name collisions: {summary['name_collisions_detected']}")
        logger.info(f"❌ Errors: {summary['failed_operations']}")
        
        if results.errors:
            logger.info("\nTop errors:")
            for error in results.errors[:3]:
                logger.error(f"  • {error}")
        
        if args.dry_run:
            logger.info("\n📋 DRY RUN COMPLETE - No files were renamed")
            logger.info("   Remove --dry-run to execute changes")
        else:
            logger.info(f"\n🎉 RENAMING COMPLETE!")
            if results_file:
                logger.info(f"   Results saved to: {results_file}")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Operation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    exit(main())
