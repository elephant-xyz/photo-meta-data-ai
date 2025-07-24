#!/usr/bin/env python3
"""
Script to increase timeout values in the fix script to handle slow IPFS responses.
"""

import re

def increase_timeouts_in_file(filepath):
    """Increase timeout values in the specified file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Track changes
        changes_made = []
        
        # Increase IPFS schema fetching timeout from 60 to 120 seconds
        old_timeout = 'timeout=60'
        new_timeout = 'timeout=120'
        if old_timeout in content:
            content = content.replace(old_timeout, new_timeout)
            changes_made.append(f"IPFS schema timeout: 60s → 120s")
        
        # Increase subprocess timeout from 300 to 600 seconds (5 to 10 minutes)
        old_subprocess_timeout = 'timeout=300'
        new_subprocess_timeout = 'timeout=600'
        if old_subprocess_timeout in content:
            content = content.replace(old_subprocess_timeout, new_subprocess_timeout)
            changes_made.append(f"Subprocess timeout: 300s → 600s")
        
        # Increase OpenAI timeout from 30 to 60 seconds
        old_openai_timeout = 'timeout=30  # 30 second timeout per request'
        new_openai_timeout = 'timeout=60  # 60 second timeout per request'
        if old_openai_timeout in content:
            content = content.replace(old_openai_timeout, new_openai_timeout)
            changes_made.append(f"OpenAI timeout: 30s → 60s")
        
        # Write the updated content back
        with open(filepath, 'w') as f:
            f.write(content)
        
        if changes_made:
            print(f"✅ Updated {filepath}:")
            for change in changes_made:
                print(f"   - {change}")
            return True
        else:
            print(f"⏭️  No timeout changes needed in {filepath}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {filepath}: {e}")
        return False

def main():
    """Main function to increase timeouts in relevant files."""
    print("🔧 Increasing timeout values to handle slow responses...")
    
    files_to_update = [
        "src/fix_and_submit_local.py",
        "get_layout_schema.py"
    ]
    
    total_updated = 0
    
    for filepath in files_to_update:
        if increase_timeouts_in_file(filepath):
            total_updated += 1
    
    print(f"\n📊 Summary:")
    print(f"   Total files updated: {total_updated}")
    print(f"   Total files processed: {len(files_to_update)}")
    
    if total_updated > 0:
        print("\n💡 Timeout increases applied:")
        print("   - IPFS schema fetching: 60s → 120s")
        print("   - Subprocess operations: 300s → 600s")
        print("   - OpenAI API calls: 30s → 60s")
        print("\n🔄 You can now retry your fix operation with increased timeouts.")

if __name__ == "__main__":
    main() 