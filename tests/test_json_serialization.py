#!/usr/bin/env python3
"""
Test JSON serialization of NamedTuples and map objects
"""

import json
from typing import NamedTuple

import pytest
import orjson


class SampleData(NamedTuple):
    """Sample NamedTuple for testing"""
    name: str
    value: int
    nested: dict[str, str]


class TestJSONSerialization:
    """Test JSON serialization behavior"""

    def test_namedtuple_direct_serialization(self):
        """Test if json.dumps can serialize NamedTuple directly"""
        data = SampleData(
            name="test",
            value=42,
            nested={"key": "value"}
        )
        
        # This should fail - NamedTuple is not JSON serializable
        with pytest.raises(TypeError) as exc_info:
            json.dumps(data)
        
        assert "not JSON serializable" in str(exc_info.value)

    def test_namedtuple_asdict_serialization(self):
        """Test if _asdict() makes NamedTuple serializable"""
        data = SampleData(
            name="test",
            value=42,
            nested={"key": "value"}
        )
        
        # This should work
        result = json.dumps(data._asdict())
        parsed = json.loads(result)
        
        assert parsed["name"] == "test"
        assert parsed["value"] == 42
        assert parsed["nested"]["key"] == "value"

    def test_map_object_serialization(self):
        """Test if json.dumps can serialize map objects"""
        numbers = [1, 2, 3, 4, 5]
        squared = map(lambda x: x ** 2, numbers)
        
        # This should fail - map object is not JSON serializable
        with pytest.raises(TypeError) as exc_info:
            json.dumps(squared)
        
        assert "not JSON serializable" in str(exc_info.value)

    def test_map_object_list_serialization(self):
        """Test if converting map to list makes it serializable"""
        numbers = [1, 2, 3, 4, 5]
        squared = map(lambda x: x ** 2, numbers)
        
        # This should work
        result = json.dumps(list(squared))
        parsed = json.loads(result)
        
        assert parsed == [1, 4, 9, 16, 25]

    def test_nested_namedtuple_serialization(self):
        """Test serialization of nested NamedTuples"""
        class Inner(NamedTuple):
            id: int
            data: str
        
        class Outer(NamedTuple):
            name: str
            inner: Inner
        
        data = Outer(
            name="outer",
            inner=Inner(id=1, data="inner_data")
        )
        
        # Direct serialization should fail
        with pytest.raises(TypeError):
            json.dumps(data)
        
        # Using _asdict() should work but Inner remains a NamedTuple
        with pytest.raises(TypeError):
            json.dumps(data._asdict())
        
        # Need recursive conversion
        def namedtuple_to_dict(obj):
            if hasattr(obj, '_asdict'):
                return {k: namedtuple_to_dict(v) for k, v in obj._asdict().items()}
            elif isinstance(obj, list):
                return [namedtuple_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: namedtuple_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        result = json.dumps(namedtuple_to_dict(data))
        parsed = json.loads(result)
        
        assert parsed["name"] == "outer"
        assert parsed["inner"]["id"] == 1
        assert parsed["inner"]["data"] == "inner_data"

    def test_current_implementation_issue(self):
        """Test the issue in our current implementation"""
        from src.photo_data_processor import PhotoMetadata, RootMetadata
        
        # Create sample metadata
        photo = PhotoMetadata(
            name="test.jpg",
            ipfs_url="./test.jpg",
            file_format="jpeg",
            document_type="PropertyImage",
            source_http_request=None,
            request_identifier=None,
            original_url=None
        )
        
        # This works because we use _asdict()
        assert json.dumps(photo._asdict())
        
        # But RootMetadata with map objects would fail if not converted to list
        from pathlib import Path
        link_files = map(Path, ["link1.json", "link2.json"])
        
        # This would fail if we passed map directly
        with pytest.raises(TypeError):
            root = RootMetadata(
                label="Photo",
                property_seed_has_file=map(lambda f: {"/": f"./{f}"}, link_files)
            )

    def test_orjson_with_custom_serializer(self):
        """Test orjson with our custom serializer"""
        from src.photo_data_processor import default_serializer, PhotoMetadata
        
        # Test NamedTuple serialization
        data = SampleData(
            name="test",
            value=42,
            nested={"key": "value"}
        )
        
        result = orjson.dumps(data, default=default_serializer)
        parsed = json.loads(result)  # orjson returns bytes, json.loads can parse it
        
        assert parsed["name"] == "test"
        assert parsed["value"] == 42
        assert parsed["nested"]["key"] == "value"
        
        # Test map object serialization
        numbers = [1, 2, 3]
        squared = map(lambda x: x ** 2, numbers)
        
        result = orjson.dumps({"data": squared}, default=default_serializer)
        parsed = json.loads(result)
        
        assert parsed["data"] == [1, 4, 9]
        
        # Test PhotoMetadata serialization
        photo = PhotoMetadata(
            name="test.jpg",
            ipfs_url="./test.jpg",
            file_format="jpeg",
            document_type="PropertyImage",
            source_http_request=None,
            request_identifier=None,
            original_url=None
        )
        
        result = orjson.dumps(photo, default=default_serializer)
        parsed = json.loads(result)
        
        assert parsed["name"] == "test.jpg"
        assert parsed["ipfs_url"] == "./test.jpg"
        assert parsed["file_format"] == "jpeg"
        assert parsed["document_type"] == "PropertyImage"