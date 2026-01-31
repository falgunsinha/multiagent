"""
Rebuild Matplotlib Font Cache for Isaac Sim
Run this after installing new fonts (like Linux Libertine)
"""

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import font_manager

print("="*60)
print("Rebuilding Matplotlib Font Cache...")
print("="*60)

# Rebuild font cache (compatible with newer matplotlib versions)
try:
    # Method 1: Force reload without cache
    font_manager._load_fontmanager(try_read_cache=False)
    print("✅ Font cache rebuilt successfully! (Method 1)")
except:
    try:
        # Method 2: Reinitialize font manager
        fm.fontManager.__init__()
        print("✅ Font cache rebuilt successfully! (Method 2)")
    except Exception as e:
        print(f"⚠️  Could not rebuild cache: {e}")
        print("   Continuing with existing cache...")

print("\n" + "="*60)
print("Checking for Linux Libertine fonts...")
print("="*60)

# Check if Linux Libertine is available
libertine_fonts = [f.name for f in fm.fontManager.ttflist if 'Libertine' in f.name]

if libertine_fonts:
    print(f"✅ Found {len(libertine_fonts)} Linux Libertine font(s):")
    for font in set(libertine_fonts):
        print(f"   - {font}")
else:
    print("❌ Linux Libertine NOT found!")
    print("\nPlease install Linux Libertine fonts:")
    print("1. Download from: https://sourceforge.net/projects/linuxlibertine/")
    print("2. Install all .otf files (right-click → Install for all users)")
    print("3. Restart your computer")
    print("4. Run this script again")

print("\n" + "="*60)
print("All available serif fonts:")
print("="*60)
serif_fonts = sorted(set([f.name for f in fm.fontManager.ttflist if 'serif' in f.name.lower()]))
for font in serif_fonts[:10]:  # Show first 10
    print(f"   - {font}")
if len(serif_fonts) > 10:
    print(f"   ... and {len(serif_fonts) - 10} more")

print("\n" + "="*60)
print("Current matplotlib font settings:")
print("="*60)
print(f"Font family: {plt.rcParams['font.family']}")
print(f"Serif fonts: {plt.rcParams['font.serif'][:3]}")

print("\n✅ Done!")

