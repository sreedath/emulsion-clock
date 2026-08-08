"""Shared figure style matched to Physical Review Fluids conventions
(gold standard: Soto et al., Phys. Rev. Fluids 3, 083602 (2018)).

One style for every figure: STIX serif with italic math symbols, thin black
boxed axes with inward ticks, small fonts, saturated primary marker colors,
hairline red theory/annotation lines, white background, no grids, no icons.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette (PRF-like): black ink, red accent/theory, blue and green data
INK = "#000000"
RED = "#d02020"
BLUE = "#1f4e9c"
GREEN = "#1a7a30"
GRAY = "#888888"
FILL_GRAY = "#ededed"
FILL_BLUE = "#dfe9f4"
FILL_RED = "#f7dede"

RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 10,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "lines.linewidth": 1.1,
    "lines.markersize": 4.5,
    "legend.frameon": True,
    "legend.edgecolor": "#bbbbbb",
    "legend.framealpha": 1.0,
    "legend.fancybox": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "figure.dpi": 120,
}

COL_W = 3.4    # single-column width, inches
FULL_W = 7.0   # double-column width, inches


def apply():
    plt.rcParams.update(RC)


def panel_label(ax, letter, dx=0.02, dy=0.96):
    """PRF-style panel label, e.g. (a), inside top-left."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=9, va="top", ha="left")
