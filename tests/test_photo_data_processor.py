#!/usr/bin/env python3
"""
Unit tests for photo_data_processor module using pytest
"""

import json
import uuid
from pathlib import Path
from typing import Iterator
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.photo_data_processor import (
    PhotoMetadata,
    LinkMetadata,
    RootMetadata,
    ProcessedFile,
    CIDProcessingResult,
    IMAGE_EXTENSIONS,
    is_image_file,
    is_not_hidden,
    get_file_format,
    generate_uuid_prefix,
    create_photo_metadata,
    create_link_metadata,
    create_link_reference,
    extract_link_path,
    get_all_files,
    get_image_files,
    process_photo_file,
    process_cid_directory,
    create_root_metadata,
    get_cid_directories,
    extract_link_files,
    has_link_files,
    default_serializer,
    write_json_file,
)


class TestDataStructures:
    """Test NamedTuple data structures"""

    def test_photo_metadata_creation(self):
        """Test PhotoMetadata NamedTuple creation"""
        metadata = PhotoMetadata(
            name="test.jpg",
            ipfs_url="./test/test.jpg",
            file_format="jpeg",
            document_type="PropertyImage",
            source_http_request=None,
            request_identifier=None,
            original_url=None
        )
        assert metadata.name == "test.jpg"
        assert metadata.ipfs_url == "./test/test.jpg"
        assert metadata.file_format == "jpeg"
        assert metadata.document_type == "PropertyImage"

    def test_link_metadata_creation(self):
        """Test LinkMetadata NamedTuple creation"""
        metadata = LinkMetadata(
            from_cid={"/": "cid123"},
            to_file={"/": "./metadata.json"}
        )
        assert metadata.from_cid == {"/": "cid123"}
        assert metadata.to_file == {"/": "./metadata.json"}

    def test_root_metadata_creation(self):
        """Test RootMetadata NamedTuple creation"""
        metadata = RootMetadata(
            label="Photo",
            property_seed_has_file=[{"/": "./link1.json"}, {"/": "./link2.json"}]
        )
        assert metadata.label == "Photo"
        assert len(metadata.property_seed_has_file) == 2


class TestUtilityFunctions:
    """Test utility functions"""

    def test_is_image_file(self):
        """Test image file detection"""
        assert is_image_file(Path("test.jpg"))
        assert is_image_file(Path("test.JPG"))
        assert is_image_file(Path("test.jpeg"))
        assert is_image_file(Path("test.png"))
        assert not is_image_file(Path("test.txt"))
        assert not is_image_file(Path("test.pdf"))

    def test_is_not_hidden(self):
        """Test hidden file detection"""
        assert is_not_hidden(Path("test.jpg"))
        assert is_not_hidden(Path("folder/test.jpg"))
        assert not is_not_hidden(Path(".hidden"))
        assert not is_not_hidden(Path(".DS_Store"))

    def test_get_file_format(self):
        """Test file format extraction"""
        assert get_file_format(Path("test.jpg")) == "jpeg"
        assert get_file_format(Path("test.jpeg")) == "jpeg"
        assert get_file_format(Path("test.png")) == "png"
        assert get_file_format(Path("test.txt")) is None
        assert get_file_format(Path("test.PDF")) is None

    def test_generate_uuid_prefix(self):
        """Test UUID prefix generation"""
        prefix1 = generate_uuid_prefix()
        prefix2 = generate_uuid_prefix()
        
        assert len(prefix1) == 4
        assert len(prefix2) == 4
        assert prefix1 != prefix2  # Should be different

    def test_create_photo_metadata(self):
        """Test photo metadata creation"""
        file_path = Path("/base/cid/photo.jpg")
        metadata = create_photo_metadata(file_path, "cid/photo.jpg")
        
        assert metadata.name == "photo.jpg"
        assert metadata.ipfs_url == "./cid/photo.jpg"
        assert metadata.file_format == "jpeg"
        assert metadata.document_type == "PropertyImage"

    def test_create_link_metadata(self):
        """Test link metadata creation"""
        metadata = create_link_metadata("cid123", "metadata.json")
        
        assert metadata.from_.path == "cid123"
        assert metadata.to.path == "./metadata.json"
        # Test that it serializes correctly
        assert metadata.from_.to_dict() == {"/": "cid123"}
        assert metadata.to.to_dict() == {"/": "./metadata.json"}

    def test_create_link_reference(self):
        """Test link reference creation"""
        ref = create_link_reference(Path("link.json"))
        assert ref.path == "./link.json"
        assert ref.to_dict() == {"/": "./link.json"}

    def test_extract_link_path(self):
        """Test link path extraction"""
        base_dir = Path("/base")
        extractor = extract_link_path(base_dir)
        
        processed = ProcessedFile(
            metadata_path=Path("/base/cid/meta.json"),
            link_path=Path("/base/cid/link.json")
        )
        
        result = extractor(processed)
        assert result == Path("cid/link.json")


class TestFileOperations:
    """Test file operation functions"""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory structure for testing"""
        # Create CID directories
        cid1 = tmp_path / "cid1"
        cid2 = tmp_path / "cid2"
        cid1.mkdir()
        cid2.mkdir()
        
        # Create subdirectory
        subdir = cid1 / "subdir"
        subdir.mkdir()
        
        # Create image files
        (cid1 / "photo1.jpg").touch()
        (cid1 / "photo2.png").touch()
        (subdir / "photo3.jpeg").touch()
        (cid2 / "photo4.jpg").touch()
        
        # Create non-image files
        (cid1 / "document.txt").touch()
        (cid2 / ".hidden.jpg").touch()
        
        return tmp_path

    def test_get_all_files(self, temp_dir):
        """Test getting all files from directory"""
        files = list(get_all_files(temp_dir))
        assert len(files) == 6  # All files including hidden and non-images

    def test_get_image_files(self, temp_dir):
        """Test getting only image files"""
        cid1 = temp_dir / "cid1"
        images = list(get_image_files(cid1))
        
        # Should find 3 images (photo1.jpg, photo2.png, subdir/photo3.jpeg)
        assert len(images) == 3
        assert all(is_image_file(img) for img in images)
        assert all(is_not_hidden(img) for img in images)

    def test_get_cid_directories(self, temp_dir):
        """Test getting CID directories"""
        cid_dirs = get_cid_directories(temp_dir)
        
        assert len(cid_dirs) == 2
        assert all(d.is_dir() for d in cid_dirs)
        assert {d.name for d in cid_dirs} == {"cid1", "cid2"}


class TestJSONSerialization:
    """Test JSON serialization with orjson"""
    
    def test_default_serializer_with_namedtuple(self):
        """Test default serializer handles NamedTuples"""
        metadata = PhotoMetadata(
            name="test.jpg",
            ipfs_url="./test.jpg",
            file_format="jpeg",
            document_type="PropertyImage",
            source_http_request=None,
            request_identifier=None,
            original_url=None
        )
        
        # Should convert to dict
        result = default_serializer(metadata)
        assert isinstance(result, dict)
        assert result["name"] == "test.jpg"
        assert result["file_format"] == "jpeg"
    
    def test_default_serializer_with_map(self):
        """Test default serializer handles map objects"""
        numbers = map(lambda x: x * 2, [1, 2, 3])
        result = default_serializer(numbers)
        assert result == [2, 4, 6]
    
    def test_write_json_file_formatting(self, tmp_path):
        """Test JSON files are properly formatted"""
        test_file = tmp_path / "test.json"
        data = PhotoMetadata(
            name="test.jpg",
            ipfs_url="./test.jpg",
            file_format="jpeg",
            document_type="PropertyImage",
            source_http_request=None,
            request_identifier=None,
            original_url=None
        )
        
        write_json_file(test_file, data)
        
        # Read and verify formatting
        content = test_file.read_text()
        assert "{\n" in content  # Should be indented
        assert '"name": "test.jpg"' in content  # Should have proper quotes
        
        # Verify it's valid JSON
        import json
        parsed = json.loads(content)
        assert parsed["name"] == "test.jpg"


class TestProcessingFunctions:
    """Test main processing functions"""

    @pytest.fixture
    def mock_uuid(self):
        """Mock UUID generation for consistent testing"""
        with patch('src.photo_data_processor.uuid.uuid4') as mock:
            mock.side_effect = [
                Mock(hex='abcd1234567890'),
                Mock(hex='efgh1234567890'),
                Mock(hex='ijkl1234567890'),
                Mock(hex='mnop1234567890'),
            ]
            yield mock

    def test_process_photo_file(self, tmp_path, mock_uuid):
        """Test processing a single photo file"""
        # Setup
        cid_dir = tmp_path / "cid123"
        cid_dir.mkdir()
        photo_path = cid_dir / "photo.jpg"
        photo_path.touch()
        
        # Process
        result = process_photo_file("cid123", photo_path, tmp_path)
        
        # Verify
        assert isinstance(result, ProcessedFile)
        assert result.metadata_path.exists()
        assert result.link_path.exists()
        
        # Check metadata content
        with open(result.metadata_path) as f:
            metadata = json.load(f)
        assert metadata["name"] == "photo.jpg"
        assert metadata["ipfs_url"] == "./cid123/photo.jpg"
        assert metadata["file_format"] == "jpeg"
        
        # Check link content
        with open(result.link_path) as f:
            link = json.load(f)
        assert link["from_cid"] == {"/": "cid123"}
        assert link["to_file"]["/"].startswith("./")

    def test_process_cid_directory(self, tmp_path, mock_uuid):
        """Test processing a CID directory"""
        # Setup
        cid_dir = tmp_path / "cid123"
        cid_dir.mkdir()
        (cid_dir / "photo1.jpg").touch()
        (cid_dir / "photo2.png").touch()
        (cid_dir / "document.txt").touch()
        
        # Process
        result = process_cid_directory(cid_dir, tmp_path)
        
        # Verify
        assert isinstance(result, CIDProcessingResult)
        assert result.cid == "cid123"
        assert len(result.link_files) == 2  # Only images

    def test_create_root_metadata_with_links(self):
        """Test root metadata creation with link files"""
        link_files = [
            Path("cid1/link1.json"),
            Path("cid2/link2.json"),
        ]
        
        metadata = create_root_metadata(link_files)
        
        assert metadata.label == "Photo"
        assert len(metadata.property_seed_has_file) == 2
        assert metadata.property_seed_has_file[0].path == "./cid1/link1.json"
        assert metadata.property_seed_has_file[1].path == "./cid2/link2.json"
        # Test serialization
        assert metadata.property_seed_has_file[0].to_dict() == {"/": "./cid1/link1.json"}
        assert metadata.property_seed_has_file[1].to_dict() == {"/": "./cid2/link2.json"}


class TestHelperFunctions:
    """Test helper functions"""

    def test_extract_link_files(self):
        """Test extracting link files from processing result"""
        result = CIDProcessingResult(
            cid="cid123",
            link_files=[Path("link1.json"), Path("link2.json")]
        )
        
        links = extract_link_files(result)
        assert links == [Path("link1.json"), Path("link2.json")]

    def test_has_link_files(self):
        """Test checking if result has link files"""
        result_with_files = CIDProcessingResult(
            cid="cid123",
            link_files=[Path("link1.json")]
        )
        result_without_files = CIDProcessingResult(
            cid="cid456",
            link_files=[]
        )
        
        assert has_link_files(result_with_files)
        assert not has_link_files(result_without_files)


@pytest.mark.parametrize("filename,expected", [
    ("test.jpg", "jpeg"),
    ("test.jpeg", "jpeg"),
    ("test.JPG", "jpeg"),
    ("test.png", "png"),
    ("test.PNG", "png"),
    ("test.txt", None),
    ("test.webp", None),
])
def test_get_file_format_parametrized(filename, expected):
    """Parametrized test for file format detection"""
    assert get_file_format(Path(filename)) == expected


@pytest.mark.parametrize("path,is_image", [
    ("photo.jpg", True),
    ("photo.jpeg", True),
    ("photo.png", True),
    ("photo.JPG", True),
    ("photo.webp", True),  # webp is supported
    ("document.txt", False),
    ("photo.gif", False),
])
def test_is_image_file_parametrized(path, is_image):
    """Parametrized test for image file detection"""
    assert is_image_file(Path(path)) == is_image