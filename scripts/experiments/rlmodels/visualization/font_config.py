"""
Font Configuration for All Visualizations
Sets Linux Libertine as the default font for matplotlib and plotly
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import warnings

# Linux Libertine font configuration
FONT_FAMILY = 'Linux Libertine'
FALLBACK_FONTS = ['DejaVu Serif', 'Times New Roman', 'serif']

def configure_matplotlib_fonts():
    """
    Configure matplotlib to use Linux Libertine font
    Falls back to serif fonts if Linux Libertine is not available
    """
    # Check if Linux Libertine is available
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    if FONT_FAMILY in available_fonts:
        font_to_use = FONT_FAMILY
        print(f"[FONT] Using {FONT_FAMILY}")
    else:
        # Try fallback fonts
        font_to_use = None
        for fallback in FALLBACK_FONTS:
            if fallback in available_fonts:
                font_to_use = fallback
                warnings.warn(f"Linux Libertine not found. Using {fallback} instead.")
                print(f"[FONT] Linux Libertine not available, using {fallback}")
                break
        
        if font_to_use is None:
            font_to_use = 'serif'
            warnings.warn("No preferred fonts found. Using default serif.")
            print(f"[FONT] Using default serif font")
    
    # Set matplotlib rcParams
    plt.rcParams['font.family'] = font_to_use
    plt.rcParams['font.serif'] = [FONT_FAMILY] + FALLBACK_FONTS
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    
    return font_to_use

def get_plotly_font_config():
    """
    Get plotly font configuration dict
    Returns a dict that can be used in plotly layout updates
    """
    return {
        'family': f"{FONT_FAMILY}, {', '.join(FALLBACK_FONTS)}",
        'size': 14,
        'color': '#000000'
    }

def get_plotly_title_font():
    """Get plotly title font configuration"""
    return {
        'family': f"{FONT_FAMILY}, {', '.join(FALLBACK_FONTS)}",
        'size': 18,
        'color': '#000000'
    }

def get_plotly_axis_font():
    """Get plotly axis label font configuration"""
    return {
        'family': f"{FONT_FAMILY}, {', '.join(FALLBACK_FONTS)}",
        'size': 14,
        'color': '#000000'
    }

def apply_plotly_fonts(fig):
    """
    Apply Linux Libertine fonts to a plotly figure
    
    Args:
        fig: plotly figure object
    
    Returns:
        Modified figure with fonts applied
    """
    fig.update_layout(
        font=get_plotly_font_config(),
        title_font=get_plotly_title_font(),
        xaxis=dict(
            title_font=get_plotly_axis_font(),
            tickfont=get_plotly_font_config()
        ),
        yaxis=dict(
            title_font=get_plotly_axis_font(),
            tickfont=get_plotly_font_config()
        ),
        legend=dict(
            font=get_plotly_font_config()
        )
    )
    
    # For subplots, update all xaxis and yaxis
    for i in range(1, 10):  # Support up to 9 subplots
        for axis_type in ['xaxis', 'yaxis']:
            axis_key = f'{axis_type}{i}' if i > 1 else axis_type
            if axis_key in fig.layout:
                fig.update_layout({
                    axis_key: dict(
                        title_font=get_plotly_axis_font(),
                        tickfont=get_plotly_font_config()
                    )
                })
    
    return fig

def install_linux_libertine_instructions():
    """Print instructions for installing Linux Libertine font"""
    instructions = """
    ╔════════════════════════════════════════════════════════════════╗
    ║          LINUX LIBERTINE FONT NOT FOUND                        ║
    ╚════════════════════════════════════════════════════════════════╝
    
    To install Linux Libertine font:
    
    Windows:
    1. Download from: https://sourceforge.net/projects/linuxlibertine/
    2. Extract the ZIP file
    3. Open the 'Fonts' folder
    4. Right-click on .ttf files → Install for all users
    5. Restart Python/Jupyter
    
    Linux:
    sudo apt-get install fonts-linuxlibertine
    
    macOS:
    brew install --cask font-linux-libertine
    
    After installation, restart your Python session.
    """
    print(instructions)

# Auto-configure matplotlib when module is imported
_configured_font = configure_matplotlib_fonts()

if __name__ == "__main__":
    print(f"\nConfigured font: {_configured_font}")
    print(f"Matplotlib font family: {plt.rcParams['font.family']}")
    print(f"Plotly font config: {get_plotly_font_config()}")
    
    # Check if Linux Libertine is available
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if FONT_FAMILY not in available_fonts:
        install_linux_libertine_instructions()

