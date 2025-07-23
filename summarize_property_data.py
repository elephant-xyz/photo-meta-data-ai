#!/usr/bin/env python3
"""
Property Data Summarizer
Summarizes AI analysis data including layouts, structure, lot, utility, and appliance details.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_json_file(filepath: str) -> Optional[Dict]:
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"    [!] Error loading {filepath}: {e}")
        return None


def summarize_layouts(output_dir: str) -> Dict:
    """Summarize layout data from layout JSON files."""
    layouts_summary = {
        "total_layouts": 0,
        "space_types": [],
        "details": []
    }
    
    # Look for layout files
    layout_files = list(Path(output_dir).glob("layout_*.json"))
    
    for layout_file in layout_files:
        data = load_json_file(str(layout_file))
        if data:
            layouts_summary["total_layouts"] += 1
            
            # Extract space type
            space_type = data.get("space_type", "unknown")
            if space_type and space_type not in layouts_summary["space_types"]:
                layouts_summary["space_types"].append(space_type)
            
            # Extract layout details
            layout_detail = {
                "file": layout_file.name,
                "space_type": space_type,
                "dimensions": data.get("dimensions", {}),
                "features": data.get("features", []),
                "appliances": data.get("appliances", []),
                "description": data.get("description", "")
            }
            layouts_summary["details"].append(layout_detail)
    
    return layouts_summary


def summarize_structure(output_dir: str) -> Dict:
    """Summarize structure data from structure JSON file."""
    structure_summary = {
        "file_found": False,
        "details": {}
    }
    
    structure_file = Path(output_dir) / "structure.json"
    if structure_file.exists():
        data = load_json_file(str(structure_file))
        if data:
            structure_summary["file_found"] = True
            structure_summary["details"] = {
                "building_type": data.get("building_type", "unknown"),
                "construction_material": data.get("construction_material", "unknown"),
                "roof_type": data.get("roof_type", "unknown"),
                "stories": data.get("stories", "unknown"),
                "year_built": data.get("year_built", "unknown"),
                "square_footage": data.get("square_footage", "unknown"),
                "condition": data.get("condition", "unknown"),
                "features": data.get("features", [])
            }
    
    return structure_summary


def summarize_lot(output_dir: str) -> Dict:
    """Summarize lot data from lot JSON file."""
    lot_summary = {
        "file_found": False,
        "details": {}
    }
    
    lot_file = Path(output_dir) / "lot.json"
    if lot_file.exists():
        data = load_json_file(str(lot_file))
        if data:
            lot_summary["file_found"] = True
            lot_summary["details"] = {
                "lot_size": data.get("lot_size", "unknown"),
                "lot_dimensions": data.get("lot_dimensions", "unknown"),
                "landscape_features": data.get("landscape_features", []),
                "outdoor_features": data.get("outdoor_features", []),
                "parking": data.get("parking", "unknown"),
                "zoning": data.get("zoning", "unknown")
            }
    
    return lot_summary


def summarize_utility(output_dir: str) -> Dict:
    """Summarize utility data from utility JSON file."""
    utility_summary = {
        "file_found": False,
        "details": {}
    }
    
    utility_file = Path(output_dir) / "utility.json"
    if utility_file.exists():
        data = load_json_file(str(utility_file))
        if data:
            utility_summary["file_found"] = True
            utility_summary["details"] = {
                "heating": data.get("heating", "unknown"),
                "cooling": data.get("cooling", "unknown"),
                "electrical": data.get("electrical", "unknown"),
                "plumbing": data.get("plumbing", "unknown"),
                "internet": data.get("internet", "unknown"),
                "security": data.get("security", "unknown"),
                "other_utilities": data.get("other_utilities", [])
            }
    
    return utility_summary


def summarize_appliances(output_dir: str) -> Dict:
    """Summarize appliance data from appliance JSON files."""
    appliances_summary = {
        "total_appliances": 0,
        "appliance_types": [],
        "details": []
    }
    
    # Look for appliance files
    appliance_files = list(Path(output_dir).glob("appliance_*.json"))
    
    for appliance_file in appliance_files:
        data = load_json_file(str(appliance_file))
        if data:
            appliances_summary["total_appliances"] += 1
            
            # Extract appliance type
            appliance_type = data.get("appliance_type", "unknown")
            if appliance_type and appliance_type not in appliances_summary["appliance_types"]:
                appliances_summary["appliance_types"].append(appliance_type)
            
            # Extract appliance details
            appliance_detail = {
                "file": appliance_file.name,
                "appliance_type": appliance_type,
                "brand": data.get("brand", "unknown"),
                "model": data.get("model", "unknown"),
                "condition": data.get("condition", "unknown"),
                "age": data.get("age", "unknown"),
                "features": data.get("features", []),
                "location": data.get("location", "unknown")
            }
            appliances_summary["details"].append(appliance_detail)
    
    return appliances_summary


def print_summary(property_id: str, summary: Dict):
    """Print a formatted summary of the property data."""
    print(f"\n{'='*60}")
    print(f"PROPERTY SUMMARY: {property_id}")
    print(f"{'='*60}")
    
    # Layout Summary
    print(f"\n📋 LAYOUTS ({summary['layouts']['total_layouts']} total)")
    print(f"{'-'*40}")
    if summary['layouts']['space_types']:
        # Filter out None values
        valid_space_types = [st for st in summary['layouts']['space_types'] if st and st != "unknown" and st != "None"]
        if valid_space_types:
            print(f"Space Types: {', '.join(valid_space_types)}")
        for layout in summary['layouts']['details']:
            space_type = layout.get('space_type', 'unknown')
            description = layout.get('description', 'No description')
            # Skip layouts with None space_type
            if space_type and space_type != "unknown" and space_type != "None":
                print(f"  • {space_type}: {description}")
    else:
        print("  No layout data found")
    
    # Structure Summary
    print(f"\n🏠 STRUCTURE")
    print(f"{'-'*40}")
    if summary['structure']['file_found']:
        details = summary['structure']['details']
        if details['building_type'] != "unknown":
            print(f"  Building Type: {details['building_type']}")
        if details['construction_material'] != "unknown":
            print(f"  Construction: {details['construction_material']}")
        if details['roof_type'] != "unknown":
            print(f"  Roof Type: {details['roof_type']}")
        if details['stories'] != "unknown":
            print(f"  Stories: {details['stories']}")
        if details['year_built'] != "unknown":
            print(f"  Year Built: {details['year_built']}")
        if details['square_footage'] != "unknown":
            print(f"  Square Footage: {details['square_footage']}")
        if details['condition'] != "unknown":
            print(f"  Condition: {details['condition']}")
        if details['features']:
            print(f"  Features: {', '.join(details['features'])}")
    else:
        print("  No structure data found")
    
    # Lot Summary
    print(f"\n🌳 LOT")
    print(f"{'-'*40}")
    if summary['lot']['file_found']:
        details = summary['lot']['details']
        if details['lot_size'] != "unknown":
            print(f"  Lot Size: {details['lot_size']}")
        if details['lot_dimensions'] != "unknown":
            print(f"  Dimensions: {details['lot_dimensions']}")
        if details['parking'] != "unknown":
            print(f"  Parking: {details['parking']}")
        if details['zoning'] != "unknown":
            print(f"  Zoning: {details['zoning']}")
        if details['landscape_features']:
            print(f"  Landscape: {', '.join(details['landscape_features'])}")
        if details['outdoor_features']:
            print(f"  Outdoor Features: {', '.join(details['outdoor_features'])}")
    else:
        print("  No lot data found")
    
    # Utility Summary
    print(f"\n⚡ UTILITIES")
    print(f"{'-'*40}")
    if summary['utility']['file_found']:
        details = summary['utility']['details']
        if details['heating'] != "unknown":
            print(f"  Heating: {details['heating']}")
        if details['cooling'] != "unknown":
            print(f"  Cooling: {details['cooling']}")
        if details['electrical'] != "unknown":
            print(f"  Electrical: {details['electrical']}")
        if details['plumbing'] != "unknown":
            print(f"  Plumbing: {details['plumbing']}")
        if details['internet'] != "unknown":
            print(f"  Internet: {details['internet']}")
        if details['security'] != "unknown":
            print(f"  Security: {details['security']}")
        if details['other_utilities']:
            print(f"  Other: {', '.join(details['other_utilities'])}")
    else:
        print("  No utility data found")
    
    # Appliance Summary
    print(f"\n🔌 APPLIANCES ({summary['appliances']['total_appliances']} total)")
    print(f"{'-'*40}")
    if summary['appliances']['appliance_types']:
        # Filter out None values
        valid_appliance_types = [at for at in summary['appliances']['appliance_types'] if at and at != "unknown" and at != "None"]
        if valid_appliance_types:
            print(f"Types: {', '.join(valid_appliance_types)}")
        for appliance in summary['appliances']['details']:
            appliance_type = appliance.get('appliance_type', 'unknown')
            brand = appliance.get('brand', 'unknown')
            model = appliance.get('model', 'unknown')
            condition = appliance.get('condition', 'unknown')
            
            # Only show appliance if it has meaningful data
            if appliance_type and appliance_type != "unknown" and appliance_type != "None":
                brand_info = f" - {brand}" if brand and brand != "unknown" and brand != "None" else ""
                model_info = f" {model}" if model and model != "unknown" and model != "None" else ""
                condition_info = f" ({condition})" if condition and condition != "unknown" and condition != "None" else ""
                print(f"  • {appliance_type}{brand_info}{model_info}{condition_info}")
    else:
        print("  No appliance data found")
    
    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Summarize property data from AI analysis output")
    parser.add_argument("--property-id", required=True, help="Property ID to summarize")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    
    args = parser.parse_args()
    
    # Construct the property output directory
    property_dir = os.path.join(args.output_dir, args.property_id)
    
    if not os.path.exists(property_dir):
        print(f"❌ Property directory not found: {property_dir}")
        print(f"Available properties in {args.output_dir}:")
        if os.path.exists(args.output_dir):
            for item in os.listdir(args.output_dir):
                if os.path.isdir(os.path.join(args.output_dir, item)):
                    print(f"  • {item}")
        return
    
    print(f"📊 Analyzing property data from: {property_dir}")
    
    # Generate summaries
    summary = {
        "property_id": args.property_id,
        "layouts": summarize_layouts(property_dir),
        "structure": summarize_structure(property_dir),
        "lot": summarize_lot(property_dir),
        "utility": summarize_utility(property_dir),
        "appliances": summarize_appliances(property_dir)
    }
    
    # Print the summary
    print_summary(args.property_id, summary)
    
    # Save summary to JSON file
    summary_file = os.path.join(property_dir, "property_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Summary saved to: {summary_file}")


if __name__ == "__main__":
    main() 