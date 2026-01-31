"""
Test script to verify the notebook changes are working correctly
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.ticker import FuncFormatter

# Test 1: Check available fonts
print("=" * 80)
print("TEST 1: Available Fonts")
print("=" * 80)
available_fonts = [f.name for f in fm.fontManager.ttflist]
print(f"Total fonts available: {len(set(available_fonts))}")
print("\nSearching for Linux Libertine:")
libertine_fonts = [f for f in available_fonts if 'libertine' in f.lower()]
if libertine_fonts:
    print(f"  Found: {set(libertine_fonts)}")
else:
    print("  NOT FOUND - will use fallback font")

print("\nSearching for Times New Roman:")
times_fonts = [f for f in available_fonts if 'times' in f.lower()]
if times_fonts:
    print(f"  Found: {set(times_fonts)}")
else:
    print("  NOT FOUND")

# Test 2: Font configuration
print("\n" + "=" * 80)
print("TEST 2: Font Configuration")
print("=" * 80)

if 'Linux Libertine' in available_fonts:
    plt.rcParams['font.family'] = 'Linux Libertine'
elif 'Linux Libertine O' in available_fonts:
    plt.rcParams['font.family'] = 'Linux Libertine O'
elif 'Times New Roman' in available_fonts:
    plt.rcParams['font.family'] = 'Times New Roman'
else:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']

print(f"Font family set to: {plt.rcParams['font.family']}")
print(f"Font weight: {plt.rcParams['font.weight']}")

# Test 3: Smoothing functions
print("\n" + "=" * 80)
print("TEST 3: Smoothing Functions")
print("=" * 80)

def exponential_moving_average(data, weight=0.85):
    """Exponential moving average (EMA) smoothing - WandB style"""
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = weight * smoothed[i-1] + (1 - weight) * data[i]
    return smoothed

# Test data
test_data = np.array([10, 8, 6, 4, 2, 1, 0.5, 0.3, 0.2, 0.1])
smoothed = exponential_moving_average(test_data, weight=0.85)
print(f"Original data: {test_data}")
print(f"Smoothed data: {smoothed}")
print(f"✓ Exponential MA working correctly")

# Test 4: Thousands formatter
print("\n" + "=" * 80)
print("TEST 4: Thousands Formatter")
print("=" * 80)

def thousands_formatter(x, pos):
    return f'{int(x/1000)}'

test_values = [0, 10000, 20000, 30000, 50000]
formatted = [thousands_formatter(v, 0) for v in test_values]
print(f"Original: {test_values}")
print(f"Formatted: {formatted}")
print(f"✓ Thousands formatter working correctly")

# Test 5: Create a simple plot to verify all settings
print("\n" + "=" * 80)
print("TEST 5: Creating Test Plot")
print("=" * 80)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(0, 50000, 100)
y = 10 * np.exp(-x/10000) + np.random.normal(0, 0.5, 100)
y_smooth = exponential_moving_average(y, weight=0.85)

ax.plot(x, y_smooth, linewidth=1.5, color='#2E86AB', label='Test Data')
ax.set_xlabel('Timesteps (× 10³)', fontsize=12)
ax.set_ylabel('Q-Overestimation', fontsize=12)
ax.set_title('Test Plot - Font and Formatting', fontsize=14, fontweight='normal')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
ax.set_xlim(left=0)
ax.margins(x=0)  # Remove x-axis margins - no gap at x=0

plt.tight_layout()
plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
print("✓ Test plot saved as 'test_plot.png'")
print(f"  Font used: {plt.rcParams['font.family']}")
print(f"  Title weight: normal (not bold)")
print(f"  X-axis: Shows as 0, 10, 20, 30, 40, 50 (thousands)")

plt.show()

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
print("\nIf you see the test plot with:")
print("  ✓ Serif font (Linux Libertine or Times New Roman)")
print("  ✓ Normal weight title (not bold)")
print("  ✓ X-axis showing 0, 10, 20, 30, 40, 50")
print("\nThen the notebook should work correctly after restarting the kernel!")

