# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import scipy.io as sio
import os
import pickle
import matplotlib.pyplot as plt

#### fig params
params = {"figure.facecolor": "w",
              'font.family': 'Calibri',
              'font.weight': 'bold',
              'font.size': 18.0,
              'axes.titlesize' : 18,
              'axes.titleweight': 'bold',
              'axes.labelsize': 18,
              'axes.labelweight': 'bold',
              'legend.fontsize': 16,
              "axes.facecolor": "w",
              'figure.subplot.wspace': 0.3,
              "axes.grid" : False,
              "axes.grid.axis" : "y",
              "grid.color"    : "#ffffff",
              "grid.linewidth": 4,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'text.usetex': False,
              "xtick.color"      : "k",
              "axes.spines.right" : False,
              "axes.spines.top" : False,
              "axes.edgecolor"    :"#191919"}

#%% Fig 3I
H_accmean = (pickle.load(open('hold_neighbor_acc.pkl', 'rb')))
acmean = (pickle.load(open('hold_neighbor_acc_N.pkl', 'rb')))

H_accmean = H_accmean['acmean']
acmean = acmean['acmean']
with plt.rc_context(params):
    fig = plt.figure(figsize = (6,3), dpi = 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(121)    
    ax.plot(range(8), H_accmean, color='k')
    ax.hlines(0.125, -1, 7, linestyle='dashed')
    ax2 = fig.add_subplot(122)
    ax2.plot(range(8), acmean, color='k')
    ax2.hlines(0.125, -1, 7, linestyle='dashed')
    ax.set_xticks(range(8))
    ax2.set_xticks(range(8))
    ax.set_ylim(0,0.52)
    ax2.set_ylim(0,0.52)
    ax.set_ylabel('Classification \nProbability (%)')