#!/usr/bin/env python3
"""
Integration tests for photo_data_processor module
"""

import json
from pathlib import Path

import pytest

from src.photo_data_processor import process_photo_data_group


class TestIntegration:
    """Integration tests for the complete workflow"""

    @pytest.fixture
    def sample_photo_structure(self, tmp_path):
        """Create a sample directory structure with photos"""
        # Create base directory
        photo_data_group = tmp_path / "photo_data_group"
        photo_data_group.mkdir()
        
        # Create CID directories
        cid1 = photo_data_group / "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco"
        cid2 = photo_data_group / "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
        cid1.mkdir()
        cid2.mkdir()
        
        # Create subdirectories
        subdir1 = cid1 / "interior"
        subdir2 = cid2 / "exterior"
        subdir1.mkdir()
        subdir2.mkdir()
        
        # Create image files
        (cid1 / "living_room.jpg").write_text("fake image data")
        (cid1 / "kitchen.png").write_text("fake image data")
        (subdir1 / "bedroom.jpeg").write_text("fake image data")
        (cid2 / "front_view.jpg").write_text("fake image data")
        (subdir2 / "backyard.png").write_text("fake image data")
        
        # Create non-image files (should be ignored)
        (cid1 / "description.txt").write_text("property description")
        (cid2 / ".DS_Store").write_text("hidden file")
        
        return photo_data_group

    def test_full_processing_workflow(self, sample_photo_structure):
        """Test the complete processing workflow"""
        # Process the directory
        process_photo_data_group(sample_photo_structure)
        
        # Verify root.json was created
        root_json = sample_photo_structure / "root.json"
        assert root_json.exists()
        
        # Load and verify root.json content
        with open(root_json) as f:
            root_data = json.load(f)
        
        assert root_data["label"] == "Photo"
        assert "property_seed_has_file" in root_data
        assert len(root_data["property_seed_has_file"]) == 5  # 5 images total
        
        # Verify all link files exist
        for link_ref in root_data["property_seed_has_file"]:
            link_path = sample_photo_structure / link_ref["/"].lstrip("./")
            assert link_path.exists()
            
            # Load and verify link file content
            with open(link_path) as f:
                link_data = json.load(f)
            
            assert "from" in link_data
            assert "/" in link_data["from"]
            assert "to" in link_data
            assert "/" in link_data["to"]
            
            # Verify metadata file exists
            # The 'to' field is relative to the link file location
            link_dir = link_path.parent
            metadata_path = link_dir / link_data["to"]["/"].lstrip("./")
            assert metadata_path.exists()
            
            # Load and verify metadata content
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert "name" in metadata
            assert "ipfs_url" in metadata
            assert "file_format" in metadata
            assert metadata["document_type"] == "PropertyImage"
            assert metadata["source_http_request"] is None
            assert metadata["request_identifier"] is None
            assert metadata["original_url"] is None

    def test_empty_directory(self, tmp_path):
        """Test processing an empty directory"""
        empty_dir = tmp_path / "empty_photo_data_group"
        empty_dir.mkdir()
        
        # Should not raise an error
        process_photo_data_group(empty_dir)
        
        # Root.json should not be created
        assert not (empty_dir / "root.json").exists()

    def test_no_images_directory(self, tmp_path):
        """Test processing a directory with no images"""
        no_images_dir = tmp_path / "no_images"
        no_images_dir.mkdir()
        
        cid_dir = no_images_dir / "cid123"
        cid_dir.mkdir()
        
        # Only non-image files
        (cid_dir / "document.txt").write_text("text file")
        (cid_dir / "data.csv").write_text("csv data")
        
        process_photo_data_group(no_images_dir)
        
        # Root.json should not be created
        assert not (no_images_dir / "root.json").exists()

    def test_json_files_placed_correctly(self, sample_photo_structure):
        """Test that JSON files are placed in the same directory as photos"""
        process_photo_data_group(sample_photo_structure)
        
        # Check that JSON files are in the same directories as their photos
        cid1 = sample_photo_structure / "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco"
        cid1_jsons = list(cid1.glob("*.json"))
        assert len(cid1_jsons) > 0  # Should have JSON files
        
        # Check subdirectory
        subdir1 = cid1 / "interior"
        subdir1_jsons = list(subdir1.glob("*.json"))
        assert len(subdir1_jsons) > 0  # Should have JSON files for bedroom.jpeg