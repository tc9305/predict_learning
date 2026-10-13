"""Consistent, noninteractive experiment figures."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.cache'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'figure.dpi': 120, 'axes.grid': True,
                     'grid.alpha': 0.2, 'savefig.dpi': 160})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
