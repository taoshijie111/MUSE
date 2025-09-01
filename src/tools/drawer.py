import matplotlib.pyplot as plt


def setup_nature_style(width_mm: float = 60, aspect_ratio: float = 1.5) -> None:
    """
    Set up matplotlib parameters according to Nature journal guidelines.
    
    Args:
        width_mm: Width of the figure in millimeters (default: 60)
        aspect_ratio: Aspect ratio of the figure (default: 1.5)
    """
    width_inches = width_mm / 25.4  # Convert mm to inches
    height_inches = width_inches / aspect_ratio
    
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': (width_inches, height_inches),
        'figure.dpi': 300,
        'figure.constrained_layout.use': True,
        
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 8,
        
        # Axes settings
        'axes.linewidth': 0.5,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        # 'axes.spines.right': False,
        # 'axes.spines.top': False,
        
        # Tick settings
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        
        # Grid settings
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        
        # Legend settings
        'legend.frameon': True,
        'legend.fontsize': 7,
        'legend.handlelength': 1.0,
        'legend.handletextpad': 0.5,
        'legend.borderpad': 0.2,
        'legend.columnspacing': 1.0,
        
        # Saving settings
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.dpi': 600,
        'savefig.format': 'svg'
    })

    # Set hatch density parameter globally (requires matplotlib 3.5+)
    plt.rcParams['hatch.linewidth'] = 0.3  # thinner lines
    

setup_nature_style(100)