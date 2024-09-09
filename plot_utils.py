"""Plotting helper and settings."""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


high_res_plot_settings = {
    "figure.dpi": 350,  # High resolution for print
    "savefig.dpi": 350,  # Resolution when saving figures
    "font.family": "serif",  # Serif fonts are often preferred in publications
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": [6, 4],  # Adjust for your specific needs
    "text.usetex": False,  # Set to True if you want LaTeX rendering
    "lines.linewidth": 1.5,  
    "axes.linewidth": 1.0,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "savefig.bbox": "tight",  # Ensures no excessive white space
    "savefig.format": "pdf",  # PDF is a good format for publications
}


white = np.array([255, 255, 255]) / 255

# Define various custom colormaps with a center at white
def create_custom_colormap(color1, color2, name):
  colors = np.vstack([color1, white, color2])
  return LinearSegmentedColormap.from_list(name, colors)

light_blue = np.array([173, 216, 230]) / 255
reddish_orange = np.array([255, 127, 14]) / 255
poly_cmap = create_custom_colormap(light_blue, reddish_orange,
                                         name='lightblue_white_redorange')