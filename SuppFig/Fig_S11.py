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
    
    #simres = min_max_scaler.fit_transform(simres.T)
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
    
    data = sio.loadmat(r'\Fig1\qn.mat')
    # handwood = 0
    pc1_handwood = [[],[]]
    agn = [[],[]]
    agnmean = [[],[]]
    allpc1 = [[],[]]
    handfitpc1 = [[],[]]
    handfitown = [[],[]]
    for handwood in [0,1]:    
        if handwood==0:
            pcdata = sio.loadmat(r'\Fig2\HW_sym_meanpc_hand_p15v10_2.mat') # load hand pc or wood pc   meanpc_hand_Norc3_3
        else:
            pcdata = sio.loadmat(r'\Fig2\HW_sym_meanpc_hand_p15v10_2.mat') # load hand pc or wood pc  meanpc_wood_Norc3_2
        
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
            tmp = np.array(qn[(qn[0]==i) & (qn[1]==handwood+1)][3])
            tmp = tmp[sortind]

            agnmean[handwood].append(tmp.mean())
            agn[handwood].append(tmp)
   

from statsmodels.stats.anova import AnovaRM
data = sio.loadmat(r'\Fig1\qn.mat')
color1 = ['#3A5FCD','#7EC0EE']
ag_rate = pd.DataFrame(data['qn'], columns=['sub','hw','rot','o','oc','a','ac'])
rots = np.unique(ag_rate['rot'])

ag_rate_mean = ag_rate.groupby(by=['hw','rot'], as_index=False).mean()
ag_rate_sem = ag_rate.groupby(by=['hw','rot']).sem()

mod = AnovaRM(ag_rate, 'a', 'sub', within=['hw','rot'], aggregate_func = 'mean').fit()
res = mod.anova_table

with plt.rc_context(params):
#    plt.style.use('ggplot')
    fig1 = plt.figure(figsize=(10,4), dpi=100, facecolor='w', edgecolor='k')
    ax = fig1.add_subplot(1,2,1)    
    ax.bar(rots, ag_rate_mean['a'][0:9], align='center', yerr=ag_rate_sem['a'][0:9], error_kw=dict(ecolor=color1[0], elinewidth=1),
            width=2.5,edgecolor='None', color=color1[0]) #,color=['#FF7F00','green','#007FFF']
    ax.bar(rots+2.5, ag_rate_mean['a'][9:], align='center', yerr=ag_rate_sem['a'][9:], error_kw=dict(ecolor=color1[1], elinewidth=1),
            width=2.5,edgecolor='None', color=color1[1]) #,color=['#FF7F00','green','#007FFF']
    ax.set_yticks([-3,-2,-1,0,1,2,3])  
    ax.set_xlabel('Disparity')
    ax.set_ylabel('Score') #坐标轴
    plt.show()


path = r'\SuppFig'
drift_data = sio.loadmat(path + '/hw_normalized_c3.mat')
drift_data = pd.DataFrame(drift_data['hw'])
drift_data_smallrot = drift_data[(drift_data.iloc[:,1]==1) & (drift_data.iloc[:,4].abs()>0)] # & (drift_data.iloc[:,4].abs()==10)
drift_data_smallrot.loc[:,12] = drift_data_smallrot.loc[:,12].abs().values
drift_data_smallrot.loc[:,12] = drift_data_smallrot.loc[:,12]/drift_data_smallrot.loc[:,4].abs().values
drift_data_smallrot_mean = drift_data_smallrot.groupby(by=[0], as_index=False).mean()
drift_data_smallrot_mean = drift_data_smallrot_mean[drift_data_smallrot_mean.iloc[:,0].isin(subselect)]

data = sio.loadmat(r'\Fig1\qn.mat')
own_smallrot = [np.nanmean(i[[0,1,2,3,5,6,7,8]]) for i in agn[1]]

own_smallrot = (np.array(own_smallrot)+3)/6
rho_df_own, pval_df_own = stats.pearsonr(own_smallrot,drift_data_smallrot_mean.iloc[:,-1])

with plt.rc_context(params):
    
    regr = linear_model.LinearRegression()    
    regr.fit(np.array(own_smallrot).reshape(-1,1), drift_data_smallrot_mean.iloc[:,-1].values.reshape(-1,1)) # 注意此处.reshape(-1, 1)，因为X是一维的！
    
    fig = plt.figure(figsize=(5,5),dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    l1 = ax.scatter(own_smallrot,drift_data_smallrot_mean.iloc[:,-1], facecolor='k', edgecolor='k', s=80, alpha=1)

    # xl = np.arange(0,0.9,0.1)
    # ax.plot(xl, regr.predict(np.array(xl).reshape(-1,1)), color=color1[0],linewidth=4) #peru royalblue
   
    ax.set_xlabel('Agency')
    ax.set_ylabel('Relative drift') #坐标轴
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid('off')
    ax.set_facecolor('w')
    ax.xaxis.set_label_coords(0.5, -0.16)
    ax.yaxis.set_label_coords(-0.16, 0.5)

