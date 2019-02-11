#!/usr/bin/python3

import __main__ as main
print("working on file:", main.__file__)

import matplotlib
import numpy as np

# font = {'family' : 'serif',
#         'sans-serif':['Computer Modern'],
#         'weight' : 'normal',
#         'size'   : 10}
# # {'family':'sans-serif','sans-serif':['Helvetica']}
# rc('font',**font)
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

def figsize(scale):
    fig_width_pt = 448.13095                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    # golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    # golden_mean = (np.sqrt(5.0)-1.0)/2.0  # scaled
    ratio = 0.85
    # ratio = 0.8

    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*ratio              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "axes.titlesize": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(.5),     # default fig size of 0.5 textwidth
    "figure.autolayout": True,
    'text.latex.preamble': [
        r'\usepackage[binary-units=true]{siunitx}'
        r'\DeclareSIUnit\perf{perf}',
        r'\DeclareSIUnit\load{load}',
        r'\DeclareSIUnit\store{store}',
        r'\DeclareSIUnit\size{size}',
        r'\DeclareSIUnit\flop{F}',
        r'\DeclareSIUnit\cycle{cycle}',
        r'\DeclareSIUnit\flops{FLOPS}',
        r'\DeclareSIUnit\core{core}',
        r'\DeclareSIUnit\ops{ops}'
    ],
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r'\usepackage[binary-units=true]{siunitx}'
        r'\DeclareSIUnit\perf{perf}',
        r'\DeclareSIUnit\load{load}',
        r'\DeclareSIUnit\store{store}',
        r'\DeclareSIUnit\size{size}',
        r'\DeclareSIUnit\flop{F}',
        r'\DeclareSIUnit\cycle{cycle}',
        r'\DeclareSIUnit\flops{FLOPS}',
        r'\DeclareSIUnit\core{core}',
        r'\DeclareSIUnit\ops{ops}'
    ]
}
matplotlib.rcParams.update(pgf_with_latex)
matplotlib.verbose.level = 'debug-annoying'

# colorlist = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
colorlist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# # in inches
# scaling = 1.0
# figure_width = scaling * 4
# figure_height = scaling * 3
# dims2d = (figure_width, figure_height)

markersize = 2
markersize_scatter = 4 * markersize
linewidth = 3

img_folder = 'plotted'
raw_data_folder = 'data_raw'
data_folder = 'data'
img_suffix = '.pdf'
fig_width = '0.5\\textwidth'
fig_height = '0.37\\textwidth'

devices_name_map = {
    "Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz": "Intel i7 6700k"
}
