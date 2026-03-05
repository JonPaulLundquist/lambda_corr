#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 17:59:51 2026

@author: Jon Paul Lundquist
"""
    
import numpy as np
from robust_fit import robust_fit
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

def std_ddof1(a):
    # a is the 1D array of y-values that fell in that bin
    return np.std(a, ddof=1) if a.size >= 2 else np.nan

def parabola(x, a, x0, y0):
    return a * (x - x0)**2 + y0

def parabola_fig(x, y, filename, proj='Supergalactic', grid=None, stat='Mean', varname=r'$\mathbf{\tau}$', 
                 ylim=None, title=None, bin_edges=None):

    y = np.asarray(y, float)

    if proj == 'Galactic':
        if grid is not None:
            grid_gal = grid.transform_to('galactic')
            x = np.asarray(grid_gal.b.deg)
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]
        else:
            raise ValueError("Astropy grid must be supplied for Galactic projection")
    elif proj != 'Supergalactic':
        raise ValueError("Only supergalactic and galactic projections implemented")
    
    x = np.asarray(x, float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
        
    if bin_edges is None:
        percentiles = np.linspace(0, 100, 21)
        bin_edges = np.percentile(x, percentiles)

    n, _, _      = binned_statistic(x, y, statistic="count", bins=bin_edges)
    x_mean, _, _ = binned_statistic(x, x, statistic="mean",  bins=bin_edges)
    y_std, _, _  = binned_statistic(x, y, statistic=std_ddof1,   bins=bin_edges)
    y_err = y_std / np.sqrt(np.maximum(n, 1))
    
    if stat == 'Mean':
        y_stat, _, _ = binned_statistic(x, y, statistic="mean",  bins=bin_edges)
        
    elif stat == 'Median':
        y_stat, _, _ = binned_statistic(x, y, statistic="median",  bins=bin_edges)
        y_err = 1.253*y_err
        
    else:
        print('Not Implemented')
        
    good = np.isfinite(x_mean) & np.isfinite(y_stat) & (n > 0)
    popt, score = robust_fit(parabola, x_mean[good], y_stat[good])
    a, x0, y0 = popt

    x_fit = np.linspace(-90, 90, 400)
    y_fit = parabola(x_fit, a, x0, y0)

    label = f"{stat} {varname}"
    
    s = f"{a:0.3e}"
    mant, exp = s.split("e")
    exp = int(exp)
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(x_mean[good], y_stat[good], yerr=y_err[good], fmt='s', capsize=5, linestyle='-',
                 color='blue', label=label)
    line1 = "Parabola Fit:"
    #line2 = rf"a=$\mathbf{{{mant}\times 10^{{{exp}}}}}$"
    
    aval = a*(180/np.pi)**2
    line1 = "Parabola Fit:"
    #line2 = rf"$a={aval:.3f}\ \mathrm{{rad}}^{{-2}}$"
    line2 = rf"$\mathbf{{a={aval:.3f}\ rad^{{-2}}}}$"
    plt.plot(x_fit, y_fit, color="red", label=line1 + "\n" + line2)
    plt.xlim((-90, 90))
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel(f"{proj} Latitude  [deg]", fontweight='semibold', size=18)
    plt.ylabel(label, fontweight='semibold', size=18)
    plt.grid(True, which="major", ls="--")
    ax = plt.gca()  # Get the current axes
    ticks = np.arange(-90, 91, 30)  # Generates ticks from -90 to 90 in steps of 30
    tlabels = [str(tick) for tick in ticks]  # Convert each tick value to a string

    ax.set_xticks(ticks)  # Set the ticks on the x-axis
    ax.set_xticklabels(tlabels)  # Set the labels for these ticks
    for tlabel in ax.get_xticklabels() + ax.get_yticklabels():
        tlabel.set_fontweight('semibold')

    # Increase line width of the plot box
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    # Set the font size for x-axis and y-axis tick labels
    tick_label_size = 14  # Set the desired size
    for tlabel in ax.get_xticklabels():
        tlabel.set_fontsize(tick_label_size)
    for tlabel in ax.get_yticklabels():
        tlabel.set_fontsize(tick_label_size)

    ax.xaxis.grid(True, which='both', ls="-")  # Major gridlines for x-axis
    ax.yaxis.grid(True, which='both', ls="-")   # Both major and minor gridlines for y-axis
    
    plt.title(title, y=1.04, fontweight='semibold', size=18)
    plt.legend(prop={'weight': 'semibold', 'size': 15})
    #plt.tight_layout()
    plt.savefig(filename + ".png", dpi=600)

    resid = y_stat[good] - parabola(x_mean[good], a, x0, y0)
    chi_squared = np.sum((resid / y_err[good]) ** 2)
    reduced_chi2 = chi_squared / (good.sum() - 3)
    
    return popt, score, reduced_chi2, bin_edges