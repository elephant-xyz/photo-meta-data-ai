#!/usr/bin/env python3
"""
Photo Data Processor - Functional Style
Traverses photo_data_group directory and generates JSON metadata files for images only
"""

import json
import uuid
from itertools import chain
from pathlib import Path
from typing import Any, Iterator, NamedTuple


class PhotoMetadata(NamedTuple):
    """Metadata for a single photo file"""

    name: str
    ipfs_url: str
    file_format: str | None
    document_type: str
    source_http_request: None
    request_identifier: None
    original_url: None


class LinkMetadata(NamedTuple):
    """Link between CID and photo metadata file"""

    from_cid: dict[str, str]
    to_file: dict[str, str]


class RootMetadata(NamedTuple):
    """Root metadata structure"""

    label: str
    property_seed_has_file: list[dict[str, str]]


class ProcessedFile(NamedTuple):
    """Result of processing a single photo file"""

    metadata_path: Path
    link_path: Path


class CIDProcessingResult(NamedTuple):
    """Result of processing a CID directory"""

    cid: str
    link_files: list[Path]


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_image_file(file_path: Path) -> bool:
    """Check if file is an image based on extension"""
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def get_file_format(file_path: Path) -> str | None:
    """Extract file format from file extension"""
    extension = file_path.suffix.lower().lstrip(".")
    # Normalize jpeg to jpg
    if extension == "jpg":
        return "jpeg"
    return extension if extension in {"jpeg", "png"} else None


def generate_uuid_prefix() -> str:
    """Generate first 4 characters of a random UUID"""
    return str(uuid.uuid4())[:4]


def create_photo_metadata(file_path: Path, relative_path: str) -> PhotoMetadata:
    """Create PhotoMetadata for a given file"""
    return PhotoMetadata(
        name=file_path.name,
        ipfs_url=f"./{relative_path}",
        file_format=get_file_format(file_path),
        document_type="PropertyImage",
        source_http_request=None,
        request_identifier=None,
        original_url=None,
    )


def create_link_metadata(cid: str, metadata_file_path: str) -> LinkMetadata:
    """Create LinkMetadata between CID and metadata file"""
    return LinkMetadata(from_cid={"/": cid}, to_file={"/": f"./{metadata_file_path}"})


def write_json_file(file_path: Path, data: Any) -> None:
    """Write data to JSON file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def process_photo_file(cid: str, photo_path: Path, base_dir: Path) -> ProcessedFile:
    """Process a single photo file and return paths to created files"""
    # Generate unique filename prefix
    uuid_prefix = generate_uuid_prefix()

    # Create file paths in the same directory as the photo
    photo_dir = photo_path.parent
    metadata_filename = f"{uuid_prefix}.json"
    link_filename = f"{uuid_prefix}-link.json"
    metadata_path = photo_dir / metadata_filename
    link_path = photo_dir / link_filename

    # Calculate relative path from base directory for the photo
    relative_photo_path = photo_path.relative_to(base_dir)

    # Create metadata
    photo_metadata = create_photo_metadata(photo_path, str(relative_photo_path))

    # Calculate relative path for metadata file from base directory
    relative_metadata_path = metadata_path.relative_to(base_dir)
    link_metadata = create_link_metadata(cid, str(relative_metadata_path))

    # Write files
    write_json_file(metadata_path, photo_metadata._asdict())
    write_json_file(link_path, link_metadata._asdict())

    return ProcessedFile(metadata_path=metadata_path, link_path=link_path)


def get_image_files(directory: Path) -> Iterator[Path]:
    """Recursively find all image files in directory"""
    for item in directory.rglob("*"):
        if item.is_file() and not item.name.startswith(".") and is_image_file(item):
            yield item


def process_cid_directory(cid_dir: Path, base_dir: Path) -> CIDProcessingResult:
    """Process all images in a CID directory"""
    cid = cid_dir.name

    # Process all image files
    processed_files = [process_photo_file(cid, photo_path, base_dir) for photo_path in get_image_files(cid_dir)]

    # Extract link file paths relative to base directory
    link_files = [processed.link_path.relative_to(base_dir) for processed in processed_files]

    return CIDProcessingResult(cid=cid, link_files=link_files)


def create_root_metadata(link_files: list[Path]) -> RootMetadata:
    """Create root metadata from link files"""
    return RootMetadata(label="Photo", property_seed_has_file=[{"/": f"./{link_file}"} for link_file in link_files])


def process_photo_data_group(input_dir: Path) -> None:
    """Main processing function"""
    if not input_dir.exists():
        raise ValueError(f"Input directory '{input_dir}' does not exist.")

    # Get all CID directories
    cid_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not cid_dirs:
        print(f"No CID directories found in {input_dir}")
        return

    # Process each CID directory and collect results
    processing_results = [process_cid_directory(cid_dir, input_dir) for cid_dir in cid_dirs]

    # Collect all link files
    all_link_files = list(chain.from_iterable(result.link_files for result in processing_results))

    if not all_link_files:
        print("No image files found in any CID directories")
        return

    # Create and write root metadata in the base directory
    root_metadata = create_root_metadata(all_link_files)
    root_path = input_dir / "root.json"
    write_json_file(root_path, root_metadata._asdict())

    # Print summary
    print("Processing complete:")
    print(f"  - Processed {len(cid_dirs)} CID directories")
    print(f"  - Generated {len(all_link_files)} image metadata files")
    print(f"  - Root metadata written to: {root_path}")
    print(f"  - Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")

    for result in processing_results:
        if result.link_files:
            print(f"  - CID {result.cid}: {len(result.link_files)} images")


def main() -> None:
    """Entry point for the photo data processor"""
    import argparse

    parser = argparse.ArgumentParser(description="Process photo data and generate JSON metadata files for images only")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("photo_data_group"),
        help="Input directory containing CID subdirectories (default: photo_data_group)",
    )

    args = parser.parse_args()

    try:
        process_photo_data_group(args.input_dir)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
