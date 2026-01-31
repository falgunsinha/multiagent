"""
Batch update all visualization files to use Linux Libertine font
This script adds font configuration imports to all visualization files
"""

from pathlib import Path
import re

# Font import code for matplotlib/seaborn files
MATPLOTLIB_FONT_IMPORT = """import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from font_config import configure_matplotlib_fonts
configure_matplotlib_fonts()  # Apply Linux Libertine font
"""

# Font import code for plotly files  
PLOTLY_FONT_IMPORT = """import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from font_config import apply_plotly_fonts
"""

def update_seaborn_file(file_path):
    """Add font configuration to a seaborn visualization file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already updated
    if 'font_config' in content:
        print(f"  [SKIP] {file_path.name} - already updated")
        return False
    
    # Find the imports section (after docstring, before first function/class)
    # Insert after the last import statement
    import_pattern = r'(import\s+\w+.*\n(?:from\s+\w+.*\n)*)'
    
    # Find last import
    imports = list(re.finditer(r'^(?:import|from)\s+.*$', content, re.MULTILINE))
    if imports:
        last_import = imports[-1]
        insert_pos = last_import.end()
        
        # Insert font config after last import
        new_content = (
            content[:insert_pos] + 
            '\n' + MATPLOTLIB_FONT_IMPORT + '\n' +
            content[insert_pos:]
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  [UPDATED] {file_path.name}")
        return True
    
    print(f"  [ERROR] Could not find imports in {file_path.name}")
    return False

def update_plotly_file(file_path):
    """Add font configuration to a plotly visualization file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already updated
    if 'font_config' in content:
        print(f"  [SKIP] {file_path.name} - already updated")
        return False
    
    # Find last import
    imports = list(re.finditer(r'^(?:import|from)\s+.*$', content, re.MULTILINE))
    if imports:
        last_import = imports[-1]
        insert_pos = last_import.end()
        
        # Insert font config after last import
        new_content = (
            content[:insert_pos] + 
            '\n' + PLOTLY_FONT_IMPORT + '\n' +
            content[insert_pos:]
        )
        
        # Also add apply_plotly_fonts() calls before fig.write_html() or fig.show()
        # Find all fig.write_html or fig.show calls and add apply_plotly_fonts before them
        new_content = re.sub(
            r'(\s+)(fig\.write_html|fig\.show)',
            r'\1fig = apply_plotly_fonts(fig)\n\1\2',
            new_content
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  [UPDATED] {file_path.name}")
        return True
    
    print(f"  [ERROR] Could not find imports in {file_path.name}")
    return False

def main():
    """Update all visualization files"""
    print("\n" + "="*80)
    print("UPDATING ALL VISUALIZATION FILES WITH LINUX LIBERTINE FONT")
    print("="*80 + "\n")
    
    viz_root = Path(__file__).parent
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    # Update individual models - seaborn
    print("Individual Models - Seaborn:")
    seaborn_ind = viz_root / "individual_models" / "seaborn"
    for file in seaborn_ind.glob("*.py"):
        if file.name != "__init__.py":
            result = update_seaborn_file(file)
            if result:
                updated_count += 1
            elif 'font_config' in file.read_text():
                skipped_count += 1
            else:
                error_count += 1
    
    # Update individual models - plotly
    print("\nIndividual Models - Plotly:")
    plotly_ind = viz_root / "individual_models" / "plotly"
    for file in plotly_ind.glob("*.py"):
        if file.name != "__init__.py":
            result = update_plotly_file(file)
            if result:
                updated_count += 1
            elif 'font_config' in file.read_text():
                skipped_count += 1
            else:
                error_count += 1
    
    # Update cross-model - seaborn
    print("\nCross-Model - Seaborn:")
    seaborn_cross = viz_root / "cross_model" / "seaborn"
    for file in seaborn_cross.glob("*.py"):
        if file.name != "__init__.py":
            result = update_seaborn_file(file)
            if result:
                updated_count += 1
            elif 'font_config' in file.read_text():
                skipped_count += 1
            else:
                error_count += 1
    
    # Update cross-model - plotly
    print("\nCross-Model - Plotly:")
    plotly_cross = viz_root / "cross_model" / "plotly"
    for file in plotly_cross.glob("*.py"):
        if file.name != "__init__.py":
            result = update_plotly_file(file)
            if result:
                updated_count += 1
            elif 'font_config' in file.read_text():
                skipped_count += 1
            else:
                error_count += 1
    
    print("\n" + "="*80)
    print(f"SUMMARY:")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print("="*80 + "\n")
    
    print("[INFO] All visualization files now use Linux Libertine font!")
    print("[INFO] Run any visualization script to see the updated fonts.")

if __name__ == "__main__":
    main()

