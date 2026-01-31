#!/usr/bin/env python3
"""
Script to automatically update imports in copied Isaac Sim files
to use local project structure instead of Isaac Sim modules.

Usage:
    python update_imports.py
"""

import re
from pathlib import Path

# Define import replacements
REPLACEMENTS = [
    # Gripper imports
    (
        r'from isaacsim\.robot\.manipulators\.grippers\.gripper import Gripper',
        'from src.grippers.gripper import Gripper'
    ),
    (
        r'from isaacsim\.robot\.manipulators\.grippers\.parallel_gripper import ParallelGripper',
        'from src.grippers.parallel_gripper import ParallelGripper'
    ),
    (
        r'from isaacsim\.robot\.manipulators\.grippers\.surface_gripper import SurfaceGripper',
        '# from isaacsim.robot.manipulators.grippers.surface_gripper import SurfaceGripper  # Not copied to project'
    ),
    
    # Controller imports
    (
        r'import isaacsim\.robot\.manipulators\.controllers as manipulators_controllers',
        'from src.controllers import pick_place_controller as manipulators_controllers'
    ),
    (
        r'from isaacsim\.robot\.manipulators\.controllers import PickPlaceController',
        'from src.controllers import PickPlaceController'
    ),
    
    # Manipulator imports
    (
        r'from isaacsim\.robot\.manipulators import SingleManipulator',
        'from src.manipulators import SingleManipulator'
    ),
    (
        r'from isaacsim\.robot\.manipulators\.manipulators\.single_manipulator import SingleManipulator',
        'from src.manipulators import SingleManipulator'
    ),
]

# Files to update
FILES_TO_UPDATE = [
    'src/manipulators/single_manipulator.py',
    'src/controllers/pick_place_controller.py',
    'src/controllers/franka/pick_place_controller.py',
    'src/grippers/gripper.py',
    'src/grippers/parallel_gripper.py',
]


def update_file_imports(file_path: Path, dry_run: bool = True):
    """Update imports in a single file"""
    
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return False
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {file_path}")
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    # Apply replacements
    for pattern, replacement in REPLACEMENTS:
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            changes_made.append((pattern, replacement, len(matches)))
    
    # Show changes
    if changes_made:
        print(f"  ✅ Found {len(changes_made)} import(s) to update:")
        for pattern, replacement, count in changes_made:
            print(f"     • {pattern}")
            print(f"       → {replacement}")
            if count > 1:
                print(f"       ({count} occurrences)")
    else:
        print("  ℹ️  No imports to update")
        return False
    
    # Write changes if not dry run
    if not dry_run:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  💾 File updated!")
    else:
        print(f"  ⏸️  Dry run - no changes written")
    
    return True


def main():
    """Main function"""
    
    print("=" * 70)
    print("Isaac Sim Import Updater for Cobot Project")
    print("=" * 70)
    
    # Get project root
    project_root = Path(__file__).parent
    print(f"\nProject root: {project_root}")
    
    # Ask user for confirmation
    print("\nThis script will update imports in the following files:")
    for file_path in FILES_TO_UPDATE:
        full_path = project_root / file_path
        status = "✅ Found" if full_path.exists() else "❌ Not found"
        print(f"  {status}: {file_path}")
    
    print("\n" + "=" * 70)
    print("DRY RUN MODE - No files will be modified")
    print("=" * 70)
    
    # Dry run first
    files_updated = 0
    for file_path in FILES_TO_UPDATE:
        full_path = project_root / file_path
        if update_file_imports(full_path, dry_run=True):
            files_updated += 1
    
    print("\n" + "=" * 70)
    print(f"Dry run complete: {files_updated} file(s) would be updated")
    print("=" * 70)
    
    # Ask for confirmation
    response = input("\nDo you want to apply these changes? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\n" + "=" * 70)
        print("APPLYING CHANGES")
        print("=" * 70)
        
        files_updated = 0
        for file_path in FILES_TO_UPDATE:
            full_path = project_root / file_path
            if update_file_imports(full_path, dry_run=False):
                files_updated += 1
        
        print("\n" + "=" * 70)
        print(f"✅ Complete: {files_updated} file(s) updated successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Review the changes in each file")
        print("2. Test the imports by running your script")
        print("3. Check for any remaining import errors")
        
    else:
        print("\n❌ Changes cancelled - no files were modified")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

