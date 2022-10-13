# -*- coding: utf-8 -*-


import scipy.io as sio  
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from matplotlib import gridspec
from statsmodels.stats.anova import AnovaRM


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
#%% fig.4E 

H = pickle.load(open(r'\Fig4\H_pc_pc1.pkl','rb'))
N = pickle.load(open(r'\Fig4\N_pc_pc1.pkl','rb'))
select_H = np.arange(28)

df_pc = H[0].append(N[0])
stats.ttest_rel(df_pc['hand'], df_pc['wood'])

colors = ['#3A5FCD', '#7EC0EE']
colorpc = ['#53868B', '#7AC5CD']
with plt.rc_context(params):
    f_subhw = plt.figure(figsize=(3.5, 3.2), dpi=100, facecolor='w', edgecolor='k')
    ax = f_subhw.add_subplot(1,1,1)
#    ax.scatter(thand[:,0], twood[:,0],facecolor = 'gray', edgecolor='k',marker='o', alpha=0.9, s=70)
    ax.plot([0,1.2],[0,1.2],'k', linewidth=2)
    fh = ax.scatter(df_pc.iloc[0:len(select_H),0], df_pc.iloc[0:len(select_H),1],facecolor = '#333333', edgecolor='k',marker='o', alpha=0.9, s=20)
    fn = ax.scatter(df_pc.iloc[len(select_H):,0], df_pc.iloc[len(select_H):,1],facecolor = '#333333', edgecolor='k',marker='^', alpha=0.9, s=20)
    ax.set_xlim(-0.1,1.3)
    ax.set_xticks([0,0.5,1])
    ax.set_ylim(-0.1,1.3)
    ax.set_xlabel('Arm')
    ax.set_ylabel('Wood')
    plt.tight_layout()
    plt.show()
    
    
    #########
    fig = plt.figure(figsize=(1.6, 1.5),dpi=100, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    ax.bar([0,0.5], df_pc.mean(), align='center',
            yerr = df_pc.sem(), error_kw=dict(ecolor='black',elinewidth=1),width=0.3,edgecolor = "none",
            color=[colorpc[t] for t in range(2)])
    ax.set_xticks([0,0.5])
    ax.set_yticks([0,0.4,.8])
    ax.set_xticklabels(['Arm','Wood'], fontsize=11)
    ax.set_yticklabels(['0','0.4','0.8'], fontsize=11)
    plt.show()
    
#%% 
hpc1 = H[1]
npc1 = N[1]

npc1['neuron'] += 100
df_pc1 = hpc1.append(npc1)
pc1_mean = df_pc1.groupby(by=['hand_wood','rotation'], as_index = False).mean()
pc1_sem = df_pc1.groupby(by=['hand_wood','rotation'], as_index = False).sem()

with plt.rc_context(params):
    fig = plt.figure(figsize=(4.2,4),dpi=100, facecolor='w', edgecolor='k')
    ax = plt.subplot(1,1,1)
    
    ax.errorbar(pc1_mean['rotation'][pc1_mean['hand_wood']==0],pc1_mean['pc1'][pc1_mean['hand_wood']==0], yerr = pc1_sem['pc1'][pc1_mean['hand_wood']==0].to_numpy(), color=colors[0], fmt='-o')
    ax.errorbar(pc1_mean['rotation'][pc1_mean['hand_wood']==1],pc1_mean['pc1'][pc1_mean['hand_wood']==1], yerr = pc1_sem['pc1'][pc1_mean['hand_wood']==1].to_numpy(), color=colors[1], fmt='-o')

    ax.set_xlim([-51,51])
    ax.set_ylim([0.4,1.05])
    ax.set_ylabel('P(C=1|Data)')
    ax.set_xlabel('Disparity')
    
mod = AnovaRM(df_pc1, 'pc1', 'neuron', within=['hand_wood', 'rotation'], aggregate_func = 'mean').fit()
res = mod.anova_table
plt.show()
  
#%% plot distribution of HWHM fig.4E 
from scipy import optimize

def fmax(x,n):
    return (((1 + np.cos(x))/2)**n)

color2 = ['#8B0000','#EEA9B8']
xl = np.arange(-np.pi,np.pi,0.01)
threshold = np.zeros((len(df_pc1['neuron'].unique()),2)) + np.nan
ii = 0
for i in df_pc1['neuron'].unique().astype(int):
    dat = df_pc1[df_pc1['neuron']==i]
    y = dat[dat['hand_wood']==0]['pc1']
    rotation = dat['rotation'].unique()/90*90/180*np.pi
    popt, pcov = optimize.curve_fit(fmax,rotation,y,[1])
#        popt, pcov = optimize.curve_fit(fmax,rotation,ownership,[[1],[1]], maxfev=2000)
    yhat = fmax(xl,popt[0])
#        yhat = fmax(xl,popt[0],popt[1])
    t1 = xl[(np.max(np.nonzero(yhat>0.5)))]
    t2 = xl[(np.min(np.nonzero(yhat>0.5)))]
    threshold[ii,0] = ((t1-t2)/np.pi*180/90*90/2)
    
    dat = df_pc1[df_pc1['neuron']==i]
    y = dat[dat['hand_wood']==1]['pc1']
    rotation = dat['rotation'].unique()/90*90/180*np.pi
    popt, pcov = optimize.curve_fit(fmax,rotation,y,[1])
#        popt, pcov = optimize.curve_fit(fmax,rotation,ownership,[[1],[1]], maxfev=2000)
    yhat = fmax(xl,popt[0])
#        yhat = fmax(xl,popt[0],popt[1])
    t1 = xl[(np.max(np.nonzero(yhat>0.5)))]
    t2 = xl[(np.min(np.nonzero(yhat>0.5)))]
    threshold[ii,1] = ((t1-t2)/np.pi*180/90*90/2)
    ii += 1

neuofH = np.where(df_pc1['neuron'].unique().astype(int)<100)
neuofN = np.where(df_pc1['neuron'].unique().astype(int)>=100)

with plt.rc_context(params):
    f_hw_dis = plt.figure(figsize=(3.5, 3.2),dpi=100) 
    ax = f_hw_dis.add_subplot(111)
    ax.plot([0,140],[0,140],'k')
    l1 = ax.scatter(threshold[neuofH,0], threshold[neuofH,1], facecolor='#333333', edgecolor='k', marker='o', s=20, alpha=0.9)
    l2 = ax.scatter(threshold[neuofN,0], threshold[neuofN,1], facecolor='#333333', edgecolor='k', marker='^', s=20, alpha=0.9)
    ax.set_xlim(20,130)
    ax.set_ylim(20,130)
    ax.set_xticks([50, 100])
    ax.set_yticks([50, 100])
    ax.set_xlabel('Arm')
    ax.set_ylabel('Wood')
#    plt.legend([l1, l2],['Monkey H', 'Monkey N'], bbox_to_anchor=(0.45, 0.4),framealpha=False, prop={'size': 14})
    plt.tight_layout()
    # plt.savefig(r'E:\Wen_projects\Multisensory_integration\MI_manuscript\figures\HN_pc1_dis69_2.eps',dpi=300, bbox_inches='tight')
    plt.show()
    
    ########
with plt.rc_context(params):
    fig = plt.figure(figsize=(1.6, 1.5),dpi=100, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    ax.bar([0,0.5], [threshold[:,0].mean(),threshold[:,1].mean()], align='center',
            yerr = stats.sem(threshold), error_kw=dict(ecolor='black',elinewidth=1),width=0.3,edgecolor = "none",
            color=[color2[t] for t in range(2)])
    ax.set_xticks([0,0.5])
    ax.set_yticks([0,30,60])
    ax.set_xticklabels(['Arm','Wood'], fontsize=11)
    ax.set_yticklabels(['0','30','60'], fontsize=11)
#    ax.set_ylabel('P index', fontsize=11)
    plt.show()
    

