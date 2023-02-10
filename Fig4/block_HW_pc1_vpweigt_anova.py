# -*- coding: utf-8 -*-


import scipy.io as sio  
import os
import sys
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm   
from scipy import io as spio 
from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize
from matplotlib import gridspec
import seaborn as sns
import pickle


#%%
colormap = list(sns.color_palette("muted", 3, desat=0.8))
colormap.reverse()
params = {"figure.facecolor": "w",
              'figure.autolayout': True,
              'font.family': 'Calibri',
              'font.weight': 'bold',
              'font.size': 18.0,
              'lines.linewidth': 2,
              'axes.labelsize': 18,
              'axes.labelweight': 'bold',
              'legend.fontsize': 18,
              "axes.facecolor": "w",
              'figure.subplot.wspace': 0.3,
              "axes.grid" : False,
              "axes.grid.axis" : "y",
              "grid.color"    : "#ffffff",
              "grid.linewidth": 4,
              "axes.spines.left" : True,
              "axes.spines.right" : False,
              "axes.spines.top" : False,
              "xtick.color"      : "#191919",
              "axes.edgecolor"    :"#191919",
              "axes.prop_cycle" : plt.cycler('color', colormap)}        
#%%               
def fmax(x,n,b):
    return (((1 + np.cos(x))/2)**n)*b
    
#%% prior fig 4D
pc_data = sio.loadmat(r'\Fig4\HW_sym_meanpc_hand_p15v10_2.mat')
df_pc_all = pd.DataFrame(np.hstack((pc_data['meanpc'][0][0], pc_data['meanpc'][0][1])), columns = ['hand', 'wood'])
df_pc = df_pc_all

neu_num = len(df_pc)
rotation = np.array([-45,-35,-20,-10,0,10,20,35,45])
#rotation = np.array([-35,-20,-10,0,10,20,35])
rot_num = len(rotation)

with plt.rc_context(params):
    colors = ['#3A5FCD', '#7EC0EE']              

    f_subhw = plt.figure(figsize=(3.5, 3.2), dpi=100, facecolor='w', edgecolor='k')
    ax = f_subhw.add_subplot(1,1,1)
#    ax.scatter(thand[:,0], twood[:,0],facecolor = 'gray', edgecolor='k',marker='o', alpha=0.9, s=70)
    ax.scatter(df_pc['hand'], df_pc['wood'],facecolor = '#333333', edgecolor='k',marker='o', alpha=0.9, s=20)
    ax.plot([0,1.2],[0,1.2],'k', linewidth=2)
    ax.set_xlim(-0.1,1.3)
    ax.set_ylim(-0.1,1.3)
    ax.set_xticks([0,0.5,1])
    ax.set_xlabel('Arm')
    ax.set_ylabel('Wood')
    plt.tight_layout()
    plt.show()
    
    ds = plt.figure(figsize=(5,1.5),dpi=100, facecolor='w', edgecolor='k')
    ax = plt.subplot(1,1,1)
    
    ax.hist(df_pc['hand']-df_pc['wood'], color = '#333333', edgecolor='k', alpha=0.9)
    ax.plot(np.mean(df_pc['hand']-df_pc['wood']), 4.5, color='k', marker = 'v', markersize=10)
    ax.text(np.mean(df_pc['hand']-df_pc['wood'])+0.05, 5,'0.45', fontsize = 20)
    
    
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-0,5)
    plt.show()             
              
#%% fig 4D
with plt.rc_context(params):              
    fig = plt.figure(figsize=(1.6,1.5),dpi=100, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    colorpc = ['#53868B', '#7AC5CD']
    ax.bar([0,0.5], df_pc.mean(), align='center',
            yerr = df_pc.sem(), error_kw=dict(ecolor='black',elinewidth=1),width=0.3,edgecolor = "none",
            color=[colorpc[t] for t in range(2)])
    ax.set_xticks([0,0.5])
    ax.set_yticks([0,0.4,.8])
    ax.set_xticklabels(['Arm','Wood'], fontsize=11)
    ax.set_yticklabels(['0','0.4','0.8'], fontsize=11)
    plt.show()
#    ax.set_yticklabels([0, 0.1])
t_pc, p_pc = stats.ttest_rel(df_pc['hand'],df_pc['wood'])


