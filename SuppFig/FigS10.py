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
#%% 
mpl.rcdefaults()
with plt.rc_context(params):
    
    data = sio.loadmat(r'D:\Projects\MI\BCI_data\Fig1\qn.mat')
    # handwood = 0
    pc1_handwood = [[],[]]
    own = [[],[]]
    ownmean = [[],[]]
    allpc1 = [[],[]]
    handfitpc1 = [[],[]]
    handfitown = [[],[]]
    for handwood in [0,1]:    
        if handwood==0:
            pcdata = sio.loadmat(r'D:\Projects\MI\BCI_data\Fig2\HW_sym_meanpc_hand_p15v10_2.mat') # load hand pc or wood pc   meanpc_hand_Norc3_3
        else:
            pcdata = sio.loadmat(r'D:\Projects\MI\BCI_data\Fig2\HW_sym_meanpc_hand_p15v10_2.mat') # load hand pc or wood pc  meanpc_wood_Norc3_2
        
        qn = pd.DataFrame(data['qn'])
        pc1 = pcdata['meanpc1'][0][handwood]
        xl = np.arange(-np.pi,np.pi,0.01)
        result = []
        threshold = []
        result2 = []
        threshold2 = []
        
    #    color1 = ['#74a9cf','burlywood']
    #    color2 = ['#023858','#8c510a']
        color1 = ['#3A5FCD','#7EC0EE']
        color2 = ['#8B0000','#EEA9B8']
        
        fig = plt.figure(figsize=(31/2.54, 20/2.54), dpi=100, tight_layout=True)
        subselect = np.arange(1,18)
        ii = 1
        kk = 0
        handfitown[handwood] = np.zeros((len(subselect),len(xl)))
        handfitpc1[handwood] = np.zeros((len(subselect),len(xl)))
        for i in subselect:    
            rotation = (qn[(qn[0]==i) & (qn[1]==handwood + 1)][2])/90*90/180*np.pi # 1 for hand; 2 for wood
            # average of negative and positive
            sortind = np.argsort(rotation)
            rotation = np.sort(rotation)
            ownership = np.array(qn[(qn[0]==i) & (qn[1]==handwood+1)][3])
            ownership = ownership[sortind]

            ownmean[handwood].append(ownership.mean())
            own[handwood].append(ownership)
            popt, pcov = optimize.curve_fit(fmax,rotation,ownership,[1])
            result.append([popt[0],pcov[0]])
            yhat = fmax(xl,popt[0])
            t1 = xl[(np.max(np.nonzero(yhat>0.5)))]
            t2 = xl[(np.min(np.nonzero(yhat>0.5)))]
            threshold.append((t1-t2)/np.pi*180/90*90/2)
            
            ##### plot pc1 ##############
            ppc1 = pc1[i-1,:]
            allpc1[handwood].append(ppc1)
            popt2, pcov2 = optimize.curve_fit(fmax,np.sort(rotation),ppc1,[1])
            result2.append([popt2[0],pcov2[0]])
            yhat = fmax(xl,popt2[0])
            t1 = xl[(np.max(np.nonzero(yhat>0.5)))]
            t2 = xl[(np.min(np.nonzero(yhat>0.5)))]
            threshold2.append((t1-t2)/np.pi*180/90*90/2)
            kk+=1
            ii += 1
        if handwood==0:    
            thand = np.array([np.array(threshold),np.array(threshold2)]).T
            rho_hand, pval_hand = stats.pearsonr(thand[:,0],thand[:,1])
            pc1_handwood[0] = pc1
        else:
            twood = np.array([np.array(threshold),np.array(threshold2)]).T
            rho_wood, pval_wood = stats.pearsonr(twood[:,0],twood[:,1])   
            pc1_handwood[1] = pc1
        
#%% diff of own (hand - wood) vs diff of pc Figure S10
thr_o = thand[:,0] - twood[:,0]
thr_pc1 = thand[:,1] - twood[:,1]
with plt.rc_context(params):
    regr = linear_model.LinearRegression()
    regr.fit(thr_o[:,np.newaxis],thr_pc1[:,np.newaxis]) 
    xl = np.arange(-11,45,0.1)
    
    fig = plt.figure(figsize=(5,5),dpi=300, facecolor='w', edgecolor='k') 
    ax = fig.add_subplot(1,1,1)
    ax.scatter(thr_o[:,np.newaxis],thr_pc1[:,np.newaxis], facecolor='k', edgecolor='k', s=80, alpha=1)
    ax.plot(xl, regr.predict(np.array(xl).reshape(-1,1)), color=color1[0], linewidth=4) #peru royalblue
    
    ax.set_xlabel('ownership index (hand-wood)')
    ax.set_ylabel(r'$\mathbf{P_{com}}$ index (hand-wood)') #坐标轴
    
#    ax.set_xticks([40,80])
    plt.xlim([-15,63])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid('off')
    ax.set_facecolor('w')
    ax.xaxis.set_label_coords(0.5, -0.16)
    ax.yaxis.set_label_coords(-0.16, 0.5)
    plt.show()

stats.pearsonr(thr_o,thr_pc1)


