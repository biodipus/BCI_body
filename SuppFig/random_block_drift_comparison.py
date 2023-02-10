# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.io as sio
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.anova import AnovaRM


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
              
#%% Fig S9               
random = sio.loadmat(r'\SuppFig\humExp.mat')
block = sio.loadmat(r'\SuppFig\humExp_block.mat')

random = pd.DataFrame(random['reach'], columns = ['a', 'b', 'c', 'd', 'e'])
random = random[random['b']==3]
random_mean = random.groupby(['a','d'], as_index=False).mean()
random_mean = random_mean[random_mean['d']%10 == 0]
random_mean['f'] = 1
random_mean['e'] *= -1

block = block['hw']
block = block[block[:, 1]==1, :][:,[0, 1,3, 4, 12]]
block = pd.DataFrame(block, columns = ['a', 'b', 'c', 'd', 'e'])
# block = block[block['a'].isin(np.setdiff1d(np.arange(1,23), [2,4,8,16,21]))]
block_mean = block.groupby(['a','d'], as_index=False).mean()
block_mean = block_mean[np.abs(block_mean['d'])<35]
block_mean['f'] = 2
block_mean['a'] += 100
block_mean['e'] *=-1

data = random_mean.append(block_mean)
data[['d','e']] = np.abs(data[['d', 'e']])
data = data[data['d']>0]
with plt.rc_context(params):
    fig = plt.figure(figsize = (4, 4), dpi = 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    model = np.polyfit(random_mean['d'], random_mean['e'], 3)
    xi = np.arange(-30,30)
    yirand = np.polyval(model, xi)
    
    model = np.polyfit(block_mean['d'], block_mean['e'], 3)
    yiblock = np.polyval(model, xi)
    
    ax.scatter(random_mean['d'], random_mean['e'])
    ax.scatter(block_mean['d'], block_mean['e'])
#    ax.plot(np.arange(-30,31,10), random_mean.groupby('d', as_index=False).mean()['e'].as_matrix())
#    ax.plot(np.arange(-30,31,10), block_mean.groupby('d', as_index=False).mean()['e'].as_matrix())
    l1, = ax.plot(xi, yirand, linewidth = 4)
    l2, = ax.plot(xi, yiblock, linewidth = 4)
    ax.set_xlabel('Disparity (deg)')
    ax.set_ylabel('Drift (deg)')
    plt.legend([l1, l2],['Random', 'Block'], frameon=False)
    plt.tight_layout()




