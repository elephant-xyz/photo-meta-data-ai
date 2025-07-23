# Google Colab Usage Guide

## Property Data Summarizer for Google Colab

This guide shows you how to use the property summarizer script in Google Colab with a single command.

### Method 1: Direct Script Execution

1. **Upload the script to your Colab notebook:**
   ```python
   # Upload the script file
   from google.colab import files
   uploaded = files.upload()  # Select colab_summarize_property.py
   ```

2. **Run the script with a single command:**
   ```python
   # For a specific property:
   !python colab_summarize_property.py --property-id 30434108090030050
   
   # For all properties:
   !python colab_summarize_property.py --all-properties
   ```

### Method 2: Copy Script Content

1. **Copy the entire script content into a cell:**
   ```python
   # Copy the entire content of colab_summarize_property.py here
   # (The full script content)
   ```

2. **Run the script:**
   ```python
   !python colab_summarize_property.py --property-id 30434108090030050
   ```

### Method 3: Import and Use Function

1. **Copy the script content and import the functions:**
   ```python
   # Copy the script content first, then:
   from colab_summarize_property import summarize_property, summarize_all_properties
   
   # Use the functions directly
   # Single property
   summary = summarize_property("30434108090030050")
   
   # All properties
   all_summaries = summarize_all_properties()
   ```

### Available Properties

Based on your data, you can summarize these properties:
- `30434108090030050` (1605 S US HIGHWAY 1 3E, PALM BEACH GARDENS)
- `52434205310037080` (2558 GARDENS PKWY, JUPITER)

### Example Commands

```python
# Summarize a specific property
!python colab_summarize_property.py --property-id 30434108090030050

# Summarize all properties at once
!python colab_summarize_property.py --all-properties

# Summarize the second property
!python colab_summarize_property.py --property-id 52434205310037080

# With custom output directory
!python colab_summarize_property.py --property-id 30434108090030050 --output-dir /content/output

# All properties with custom output directory
!python colab_summarize_property.py --all-properties --output-dir /content/output
```

### What the Script Does

The script analyzes the AI analysis output and provides a summary of:

- **📋 LAYOUTS**: Space types and descriptions found in the property
- **🏠 STRUCTURE**: Building type, construction, roof, stories, year built, etc.
- **🌳 LOT**: Lot size, dimensions, parking, zoning, landscape features
- **⚡ UTILITIES**: Heating, cooling, electrical, plumbing, internet, security
- **🔌 APPLIANCES**: Appliance types, brands, models, conditions

### Output

The script will:
1. Print a formatted summary to the console
2. Save a detailed JSON summary to `output/{property_id}/property_summary.json`

### Notes

- The script automatically filters out "unknown" and "None" values for cleaner output
- It only shows meaningful data that was actually found in the AI analysis
- Empty sections (like Structure, Lot, Utilities) indicate no data was found for those categories 