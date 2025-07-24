#!/usr/bin/env python3
"""
Photo Data Processor - Functional Style
Traverses photo_data_group directory and generates JSON metadata files for images only
"""

import uuid
from itertools import chain
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Callable
from functools import partial
from operator import methodcaller

import orjson


class PhotoMetadata(NamedTuple):
    """Metadata for a single photo file"""

    name: str
    ipfs_url: str
    file_format: str | None
    document_type: str
    source_http_request: None
    request_identifier: None
    original_url: None


class IPLDReference(NamedTuple):
    """IPLD reference with a single '/' key pointing to a path or CID"""

    path: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format for JSON serialization"""
        return {"/": self.path}


class LinkMetadata(NamedTuple):
    """Link between CID and photo metadata file"""

    from_cid: IPLDReference
    to_file: IPLDReference


class RootMetadata(NamedTuple):
    """Root metadata structure"""

    label: str
    property_seed_has_file: list[IPLDReference]


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
    return LinkMetadata(from_cid=IPLDReference(path=cid), to_file=IPLDReference(path=f"./{metadata_file_path}"))


def default_serializer(obj: Any) -> Any:
    """Custom serializer for orjson to handle NamedTuples and other types"""
    if isinstance(obj, IPLDReference):
        # Handle IPLDReference specifically to return dict format
        return obj.to_dict()
    elif hasattr(obj, "_asdict"):
        # Handle NamedTuple
        return obj._asdict()
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        # Handle iterators (like map objects)
        return list(obj)
    # Let orjson handle the error for unsupported types
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json_file(file_path: Path, data: Any) -> None:
    """Write data to JSON file using orjson"""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize with orjson using custom default handler
    json_bytes = orjson.dumps(data, default=default_serializer, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)

    # Write bytes to file
    with open(file_path, "wb") as f:
        f.write(json_bytes)


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
    write_json_file(metadata_path, photo_metadata)
    write_json_file(link_path, link_metadata)

    return ProcessedFile(metadata_path=metadata_path, link_path=link_path)


def is_not_hidden(path: Path) -> bool:
    """Check if path is not hidden (doesn't start with .)"""
    return not path.name.startswith(".")


def get_all_files(directory: Path) -> Iterator[Path]:
    """Get all files recursively from directory"""
    return filter(methodcaller("is_file"), directory.rglob("*"))


def get_image_files(directory: Path) -> Iterator[Path]:
    """Recursively find all image files in directory"""
    return filter(lambda p: is_not_hidden(p) and is_image_file(p), get_all_files(directory))


def extract_link_path(base_dir: Path) -> Callable[[ProcessedFile], Path]:
    """Create a function to extract relative link path from ProcessedFile"""
    return lambda processed: processed.link_path.relative_to(base_dir)


def process_cid_directory(cid_dir: Path, base_dir: Path) -> CIDProcessingResult:
    """Process all images in a CID directory"""
    cid = cid_dir.name

    # Create partial function for processing files in this CID
    process_file_in_cid = partial(process_photo_file, cid)

    # Process all image files using map
    processed_files = map(lambda photo_path: process_file_in_cid(photo_path, base_dir), get_image_files(cid_dir))

    # Extract link file paths relative to base directory using map
    # We need to materialize here because we consume the iterator
    link_files = list(map(extract_link_path(base_dir), processed_files))

    return CIDProcessingResult(cid=cid, link_files=link_files)


def create_link_reference(link_file: Path) -> IPLDReference:
    """Create a reference for a link file"""
    return IPLDReference(path=f"./{link_file}")


def create_root_metadata(link_files: list[Path]) -> RootMetadata:
    """Create root metadata from link files"""
    return RootMetadata(label="Photo", property_seed_has_file=list(map(create_link_reference, link_files)))


def is_directory(path: Path) -> bool:
    """Check if path is a directory"""
    return path.is_dir()


def get_cid_directories(input_dir: Path) -> list[Path]:
    """Get all non-hidden directories from input directory"""
    return list(filter(lambda d: is_directory(d) and is_not_hidden(d), input_dir.iterdir()))


def extract_link_files(result: CIDProcessingResult) -> list[Path]:
    """Extract link files from processing result"""
    return result.link_files


def has_link_files(result: CIDProcessingResult) -> bool:
    """Check if result has any link files"""
    return bool(result.link_files)


def process_photo_data_group(input_dir: Path) -> None:
    """Main processing function"""
    if not input_dir.exists():
        raise ValueError(f"Input directory '{input_dir}' does not exist.")

    # Get all CID directories using filter
    cid_dirs = get_cid_directories(input_dir)

    if not cid_dirs:
        print(f"No CID directories found in {input_dir}")
        return

    # Process each CID directory using map
    process_cid = partial(process_cid_directory, base_dir=input_dir)
    processing_results = list(map(process_cid, cid_dirs))

    # Collect all link files using chain and map
    all_link_files = list(chain.from_iterable(map(extract_link_files, processing_results)))

    if not all_link_files:
        print("No image files found in any CID directories")
        return

    # Create and write root metadata in the base directory
    root_metadata = create_root_metadata(all_link_files)
    root_path = input_dir / "root.json"
    write_json_file(root_path, root_metadata)

    # Print summary
    print("Processing complete:")
    print(f"  - Processed {len(cid_dirs)} CID directories")
    print(f"  - Generated {len(all_link_files)} image metadata files")
    print(f"  - Root metadata written to: {root_path}")
    print(f"  - Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")

    # Print results for directories with images
    # No need to convert filter to list since we're just iterating
    for result in filter(has_link_files, processing_results):
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
