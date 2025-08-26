#!/usr/bin/env python3
"""
Photo Data Processor - Functional Style
Traverses photo_data_group directory and generates JSON metadata files for images only
"""

import uuid
from http import HTTPStatus
from pathlib import Path
from typing import Any, Iterator, NamedTuple

import orjson
import requests

session = requests.Session()

PHOTO_DATAGROUP_CID = "bafkreievgc2kirgxpphmigdrdy5r6putjax2mz6xyjcsvjhsz3zgckrr44.json"


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

    from_: IPLDReference  # Using from_ because 'from' is a Python keyword
    to: IPLDReference


class Relationships(NamedTuple):
    property_seed_has_file: list[IPLDReference]


class RootMetadata(NamedTuple):
    """Root metadata structure"""

    label: str
    relationships: Relationships


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def resolve_seed_cid(cid: str) -> str:
    response = session.get(f"https://ipfs.io/ipfs/{cid}")
    assert response.status_code == HTTPStatus.OK
    data = response.json()
    rel_cid = data["relationships"]["property_seed"]["/"]

    response = session.get(f"https://ipfs.io/ipfs/{rel_cid}")
    assert response.status_code == HTTPStatus.OK
    return response.json()["from"]["/"]


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
    return LinkMetadata(
        from_=IPLDReference(path=resolve_seed_cid(cid)), to=IPLDReference(path=f"./{metadata_file_path}")
    )


def create_link_metadata_from_parcel_id(parcel_id: str, metadata_file_path: str, property_filename: str = None) -> LinkMetadata:
    """Create LinkMetadata between parcel ID and photo metadata file"""
    # Use provided property filename or default to property_{parcel_id}.json
    if property_filename is None:
        property_filename = f"property_{parcel_id}.json"
    
    return LinkMetadata(
        from_=IPLDReference(path=f"./{property_filename}"), 
        to=IPLDReference(path=f"./{metadata_file_path}")
    )


def default_serializer(obj: Any) -> Any:
    """Custom serializer for orjson to handle NamedTuples and other types"""
    if isinstance(obj, IPLDReference):
        # Handle IPLDReference specifically to return dict format
        return obj.to_dict()
    elif isinstance(obj, LinkMetadata):
        # Handle LinkMetadata specifically to rename from_ to from
        data = obj._asdict()
        data["from"] = data.pop("from_")
        return data
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


# Removed process_photo_file - functionality moved into process_single_directory


def is_not_hidden(path: Path) -> bool:
    """Check if path is not hidden (doesn't start with .)"""
    return not path.name.startswith(".")


def get_all_files(directory: Path) -> Iterator[Path]:
    """Get all files recursively from directory"""
    return filter(lambda p: p.is_file(), directory.rglob("*"))


def get_image_files(directory: Path) -> Iterator[Path]:
    """Recursively find all image files in directory"""
    return filter(lambda p: is_not_hidden(p) and is_image_file(p), get_all_files(directory))


# Removed extract_link_path - no longer needed


def process_single_directory(directory: Path, property_filename: str = None) -> int:
    """Process a single directory and create its own root.json
    Returns the number of images processed"""
    parcel_id = directory.name

    # Get all image files in this directory
    image_files = list(get_image_files(directory))

    if not image_files:
        return 0

    # Process each image file
    link_files = []
    for photo_path in image_files:
        # Generate unique filename prefix
        uuid_prefix = generate_uuid_prefix()

        # Create file paths in the same directory as the photo
        metadata_filename = f"{uuid_prefix}.json"
        link_filename = f"{uuid_prefix}-link.json"
        metadata_path = directory / metadata_filename
        link_path = directory / link_filename

        # Create metadata - ipfs_url points to image in same directory
        photo_metadata = create_photo_metadata(photo_path, photo_path.name)

        # Create link metadata - use parcel_id instead of CID
        link_metadata = create_link_metadata_from_parcel_id(parcel_id, metadata_filename, property_filename)

        # Write files
        write_json_file(metadata_path, photo_metadata)
        write_json_file(link_path, link_metadata)

        # Add link file to list (just filename since root.json will be in same directory)
        link_files.append(link_filename)

    # Create and write root metadata in this directory
    root_references = [IPLDReference(path=f"./{link_file}") for link_file in link_files]
    root_metadata = RootMetadata(label="Photo", relationships=Relationships(property_seed_has_file=root_references))
    root_path = directory / PHOTO_DATAGROUP_CID
    write_json_file(root_path, root_metadata)

    return len(image_files)


# Removed create_link_reference and create_root_metadata - no longer needed
# as each directory creates its own root.json with direct references


def is_directory(path: Path) -> bool:
    """Check if path is a directory"""
    return path.is_dir()


def get_cid_directories(input_dir: Path) -> list[Path]:
    """Get all non-hidden directories from input directory"""
    return list(filter(lambda d: is_directory(d) and is_not_hidden(d), input_dir.iterdir()))


# Removed extract_link_files and has_link_files - no longer needed


def process_photo_data_group(input_dir: Path, property_filename: str = None) -> None:
    """Main processing function - processes each directory independently"""
    if not input_dir.exists():
        raise ValueError(f"Input directory '{input_dir}' does not exist.")

    # Get all parcel ID directories using filter
    parcel_dirs = get_cid_directories(input_dir)

    if not parcel_dirs:
        print(f"No parcel ID directories found in {input_dir}")
        return

    # Process each directory independently
    total_images = 0
    processed_dirs = 0

    for parcel_dir in parcel_dirs:
        image_count = process_single_directory(parcel_dir, property_filename)
        if image_count > 0:
            processed_dirs += 1
            total_images += image_count
            print(f"  - Processed {parcel_dir.name}: {image_count} images, root.json created")
        else:
            print(f"  - Skipped {parcel_dir.name}: no images found")

    # Print summary
    print("\nProcessing complete:")
    print(f"  - Processed {processed_dirs} directories with images")
    print(f"  - Generated {total_images} image metadata files")
    print("  - Each directory has its own root.json")
    print(f"  - Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")


def main() -> None:
    """Entry point for the photo data processor"""
    import argparse

    parser = argparse.ArgumentParser(description="Process photo data and generate JSON metadata files for images only")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output"),
        help="Input directory containing parcel ID subdirectories (default: output)",
    )
    parser.add_argument(
        "--property-filename",
        type=str,
        default=None,
        help="Custom property filename (default: property_{parcel_id}.json)",
    )

    args = parser.parse_args()

    try:
        process_photo_data_group(args.input_dir, args.property_filename)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
