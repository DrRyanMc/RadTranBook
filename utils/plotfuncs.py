
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import erf
import scipy.optimize as opt
from scipy.optimize import curve_fit
import os

# Original default plotting font (kept as `font`).
_GILL_SANS_PATH = '/Library/Fonts/GillSans.ttc'
if os.path.exists(_GILL_SANS_PATH):
    font = fm.FontProperties(family='Gill Sans', fname=_GILL_SANS_PATH, size=12)
else:
    # Fall back to the default sans-serif font on systems without Gill Sans.
    font = fm.FontProperties(family='sans-serif', size=12)

# Additional IMC/Marshak typography profile.
_IMC_SANS_STACK = [
    "Univers LT Std",
    "TeX Gyre Heros",
    "Helvetica",
    "Arial",
    "DejaVu Sans",
]
font_imc = fm.FontProperties(
    family=_IMC_SANS_STACK,
    size=12,
    variant="small-caps",
)
def hide_spines(intx=False,inty=False,cbar_ax=None, fsize=12):
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""

    # Retrieve a list of all current figures.
    figures = [x for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    if (plt.gca().get_legend()):
        plt.setp(plt.gca().get_legend().get_texts(), fontproperties=font, size=fsize) 
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # Disable ticks.
            ax.xaxis.set_ticks_position('bottom')
            #check if it is the colorbar axis
            # When you create the colorbar, save its axes:

            # Then in your loop:
            for ax in figure.canvas.figure.get_axes():
                if cbar_ax is not None and ax is cbar_ax:
                    ax.yaxis.set_ticks_position('right')
                else:
                    ax.yaxis.set_ticks_position('left')
           # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
            for label in ax.get_xticklabels() :
                label.set_fontproperties(font)
                label.set_fontsize(fsize)
            for label in ax.get_yticklabels() :
                label.set_fontproperties(font)
                label.set_fontsize(fsize)
            #ax.set_xticklabels(ax.get_xticks(), fontproperties = font)
            ax.set_xlabel(ax.get_xlabel(), fontproperties = font, fontsize=fsize)
            ax.set_ylabel(ax.get_ylabel(), fontproperties = font, fontsize=fsize)
            ax.set_title(ax.get_title(), fontproperties = font, fontsize=fsize)
            if (inty):
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
            if (intx):
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
def show(nm,a=0,b=0,cbar_ax=None, close_after=False, font_size=12):
    hide_spines(a,b,cbar_ax, fsize=font_size)
    #ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
    #plt.yticks([1,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12], labels)
    #ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
    plt.savefig(nm, dpi=600);
    if close_after:
        plt.close()
    else:
        plt.show()