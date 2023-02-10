# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
import scipy.io as sio  
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib as mpl
import seaborn as sns

#%%
#import corrstats
def fmax(x,n):
    return (((1 + np.cos(x))/2)**n)

def corr_hw(*args):
    from sklearn.metrics.pairwise import euclidean_distances
    own_t = np.array(args[0])
    allpc1_t = np.array(args[1])
#    tmp = (own[:,0:4] + own[:,-1:-5:-1])/2
#    ownsym = own
#    ownsym[:,0:4] = tmp
#    ownsym[:,-1:-5:-1] = tmp
    
    #simarray = np.concatenate([allpc1[:,0][:,np.newaxis],own], axis=1)

    res = np.zeros((17,17))
    for i in range(17):
        res[i,0:2] = stats.spearmanr(own_t[i,:], allpc1_t[i,:])
    
    return res
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

#%% plot drift; Fig supp Figure S3A
import os
params = {"figure.facecolor": "w",
              'figure.autolayout' : True,
              'font.family': 'Calibri',
              'font.weight': 'bold',
              'font.size': 18.0,
              'axes.titlesize' : 18,
              'axes.titleweight': 'bold',
              'axes.labelsize': 18,
              'axes.labelweight': 'bold',
              'legend.fontsize': 18,
              "axes.facecolor": "w",
              'figure.subplot.wspace': 0.3,
              "axes.grid" : False,
              "axes.grid.axis" : "y",
              "grid.color"    : "#ffffff",
              "lines.linewidth": 3,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'text.usetex': False,
              'mathtext.fontset': 'custom',
              "axes.spines.right" : False,
              "axes.spines.top" : False,
              "xtick.color"      : "#191919",
              "axes.edgecolor"    :"#191919"}
              
color1 = ['#3A5FCD','#7EC0EE']
color2 = ['#8B0000','#EEA9B8']              
path = r'D:\Projects\MI\BCI_data\SuppFig'
filename = 'hw_normalized_c3.mat'
data = sio.loadmat(os.path.join(path, filename))['hw']
simudata = sio.loadmat(os.path.join(path, 'HW_sym_meanpc_hand_p15v10_2.mat'))['Data'][0,0]
subselect = np.arange(17)
with plt.rc_context(params):
    fig = plt.figure(figsize=(31/2.54, 20/2.54), dpi=100, tight_layout=True)
    for i in range(len(subselect)):
        ax = fig.add_subplot(3,6,i+1)
        subdata = pd.DataFrame(data[(data[:,0]==subselect[i]+1) & (data[:,1]==1)][:,[4,12]], columns=['rot', 'drift'])
        # subdata = subdata.rename(columns={'rot':'drift', 'drift':'rot'})
        subdata['drift'] *= -1
        idx1 = (subdata['rot']<0) & (subdata['drift']<7) & (subdata['drift'].abs()<subdata['rot'].abs()+7)
        idx2 = (subdata['rot']>0) & (subdata['drift']>-7) & (subdata['drift'].abs()<subdata['rot'].abs()+7)
        idx3 = (subdata['rot']==0) & (subdata['drift'].abs()<7)
        idx = idx1 | idx2 | idx3
        subdata = subdata.loc[idx,:]
        ax.scatter(subdata.iloc[:,0], subdata.iloc[:,1], s=2, color='k')
        submean = subdata.groupby('rot', as_index=False).mean()
#        ax.plot(submean['rot'], submean['drift'])
        model = np.polyfit(submean['rot'], submean['drift'], 3)
        xi = np.arange(-35,35)
        yi = np.polyval(model, xi)
        ax.plot(xi, yi, color=color1[0])
        
        subsimu = simudata[0, subselect[i]][0,1][0:30, :]
        subsimu = np.hstack((np.repeat(np.unique(subdata['rot']), 30).reshape(-1,1), np.reshape(subsimu,(-1,1), order='F')))
        subsimu[(subsimu[:,0]<=0) & (subsimu[:,1]>7), :] = np.nan
        subsimu[(subsimu[:,0]>=0) & (subsimu[:,1]<-7), :] = np.nan
        subsimu = subsimu[~np.isnan(subsimu[:,0]),:]
        subsimu = pd.DataFrame(subsimu, columns=['rot', 'drift'])
        subsimu = subsimu.rename(columns={'rot':'drift', 'drift':'rot'})
        simumean = simudata[0, subselect[i]][0,1].mean(0)
        ax.scatter(subsimu.iloc[:,0], subsimu.iloc[:,1], s=2, color='gray', alpha=0.5)  
#        ax.plot(np.unique(subsimu['rot']), simumean, color=color2[0])
        model = np.polyfit(subsimu.iloc[:,0], subsimu.iloc[:,1], 3)
        xi = np.arange(-35,35)
        yi = np.polyval(model, xi)
        ax.plot(xi, yi, color=color2[0])
        ax.set_ylim(-39, 39)
        ax.set_title('Subject '+str(i+1),fontname = "Calibri",fontsize=18,fontweight="bold")
        ax.set_xticklabels([])
        ax.set_yticklabels([]) 
        if i+1 == 13:
            ax.set_xticks([-20, 0, 20])
            ax.set_yticks([-20, 0, 20])
            ax.set_xticklabels(['-20', '0', '20'])
            ax.set_yticklabels(['-20', '0', '20'])
            ax.set_xlabel('Disparity (deg)')
            ax.set_ylabel('Drift (deg)')
        if i+1 == 17:
            ax.legend(['Score',r'$\mathbf{P_{com}}$'],loc='upper right',bbox_to_anchor=(2.8, 1.10),framealpha=0)
