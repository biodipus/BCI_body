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
#%% Figure S3B
mpl.rcdefaults()
with plt.rc_context(params):
    
    data = sio.loadmat('qn.mat')
    # handwood = 0
    pc1_handwood = [[],[]]
    own = [[],[]]
    ownmean = [[],[]]
    allpc1 = [[],[]]
    handfitpc1 = [[],[]]
    handfitown = [[],[]]
    for handwood in [0,1]:    
        if handwood==0:
            pcdata = sio.loadmat('HW_sym_meanpc_hand_p15v10_2.mat') # load hand pc or wood pc   meanpc_hand_Norc3_3
        else:
            pcdata = sio.loadmat('HW_sym_meanpc_hand_p15v10_2.mat') # load hand pc or wood pc  meanpc_wood_Norc3_2
        
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
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
        
        for r in [3,4,5,6]:        
            qn.iloc[:,r] = min_max_scaler.fit_transform(qn.iloc[:,r].values.reshape(-1,1)).reshape(306)
              
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
    #        tmp = (ownership[0:4]+ownership[-1:-5:-1])/2

            ownmean[handwood].append(ownership.mean())
            own[handwood].append(ownership)
            popt, pcov = optimize.curve_fit(fmax,rotation,ownership,[1])
            result.append([popt[0],pcov[0]])
            ax = fig.add_subplot(3,6,ii)
            ax.scatter(rotation,ownership,color=color1[handwood],marker='o',s=40) #burlywood  74a9cf
            yhat = fmax(xl,popt[0])
            t1 = xl[(np.max(np.nonzero(yhat>0.5)))]
            t2 = xl[(np.min(np.nonzero(yhat>0.5)))]
            threshold.append((t1-t2)/np.pi*180/90*90/2)
            ax.plot(xl,fmax(xl,popt[0]),color=color1[handwood],linewidth=3) # color1[handwood]
            handfitown[handwood][ii-1,:] = fmax(xl,popt[0])
            
            ##### plot pc1 ##############
            ppc1 = pc1[i-1,:]
            allpc1[handwood].append(ppc1)
            popt2, pcov2 = optimize.curve_fit(fmax,np.sort(rotation),ppc1,[1])
            result2.append([popt2[0],pcov2[0]])
            # ax2 = fig.add_subplot(3,6,ii)
            ax.scatter(np.sort(rotation),ppc1,color=color2[handwood],marker='o',s=40) #8c510a  023858
            yhat = fmax(xl,popt2[0])
            t1 = xl[(np.max(np.nonzero(yhat>0.5)))]
            t2 = xl[(np.min(np.nonzero(yhat>0.5)))]
            threshold2.append((t1-t2)/np.pi*180/90*90/2)
            ax.plot(xl,fmax(xl,popt2[0]),color=color2[handwood],linestyle = '--',linewidth=3) # color2[handwood]
            handfitpc1[handwood][ii-1,:] = fmax(xl,popt2[0])
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            kk+=1
            ax.set_title('Subject '+str(kk), fontname = "Calibri",fontsize=18,fontweight="bold")
            ax.set_facecolor('w')
            ax.grid('off')
            plt.xticks([-1.57,0,1.57])
            plt.xlim([-np.pi, np.pi])
            plt.ylim([0,1.1])
            plt.yticks([0,0.5,1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])  
            if ii==13:
                plt.xticks([-1.57,0,1.57], ['-90','0','90'])
                plt.yticks([0,0.5,1],['0','0.5','1'])
                ax.set_xlabel('Disparity (deg)')  
                ax.set_ylabel(r'Score and $\mathbf{P_{com}}$')
            if ii==17:
                ax.legend(['Score',r'$\mathbf{P_{com}}$'],loc='upper right',bbox_to_anchor=(2.2, 1.10),framealpha=0)
            ii += 1
        plt.tight_layout(pad=0.05, w_pad=0.5, h_pad=0.5)
    #    plt.tight_layout()
        if handwood==0:    
            thand = np.array([np.array(threshold),np.array(threshold2)]).T
            rho_hand, pval_hand = stats.pearsonr(thand[:,0],thand[:,1])
            pc1_handwood[0] = pc1
        else:
            twood = np.array([np.array(threshold),np.array(threshold2)]).T
            rho_wood, pval_wood = stats.pearsonr(twood[:,0],twood[:,1])   
            pc1_handwood[1] = pc1
        
        plt.show()
#%% Figure 2E
mpl.rcdefaults()
with plt.rc_context(params):
    
    data = sio.loadmat('qn.mat')
    # handwood = 0
    pc1_handwood = [[],[]]
    own = [[],[]]
    ownmean = [[],[]]
    allpc1 = [[],[]]
    handfitpc1 = [[],[]]
    handfitown = [[],[]]
    for handwood in [0]:    
        if handwood==0:
            pcdata = sio.loadmat('HW_sym_meanpc_hand_p15v10_2.mat') # load hand pc or wood pc   meanpc_hand_Norc3_3
        else:
            pcdata = sio.loadmat('HW_sym_meanpc_hand_p15v10_2.mat') # load hand pc or wood pc  meanpc_wood_Norc3_2
        
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
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
        
        for r in [3,4,5,6]:        
            qn.iloc[:,r] = min_max_scaler.fit_transform(qn.iloc[:,r].values.reshape(-1,1)).reshape(306)

        fig = plt.figure(figsize=(3, 2), dpi=100, tight_layout=True)
        subselect = [1,6]
        ii = 1
        kk = 0
        handfitown[handwood] = np.zeros((len(subselect),len(xl)))
        handfitpc1[handwood] = np.zeros((len(subselect),len(xl)))
        kk=1
        for i in subselect:    
            rotation = (qn[(qn[0]==i) & (qn[1]==handwood + 1)][2])
            # average of negative and positive
            sortind = np.argsort(rotation)
            rotation = np.sort(rotation)
            ownership = np.array(qn[(qn[0]==i) & (qn[1]==handwood+1)][3])
            ownership = ownership[sortind]
    #        tmp = (ownership[0:4]+ownership[-1:-5:-1])/2
            ax = fig.add_subplot(1, 2, kk)
            ax.scatter(rotation,ownership,color=color1[handwood],marker='o',s=40) #burlywood  74a9cf
            ax.plot(rotation,ownership,color=color1[handwood],linewidth=3) # color1[handwood]
            
            ##### plot pc1 ##############
            ppc1 = pc1[i-1,:]
            ax.scatter(np.sort(rotation),ppc1,color=color2[handwood],marker='o',s=40) #8c510a  023858
            
            ax.plot(np.sort(rotation),ppc1,color=color2[handwood],linestyle = '--',linewidth=3) # color2[handwood]
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            kk+=1
            ax.set_title('Subject '+str(i), fontname = "Calibri",fontsize=18,fontweight="bold")

#%% ownership Pcom correlation  Fig 2J
with plt.rc_context(params):
    regr = linear_model.LinearRegression()
    t = thand
    regr.fit(t[:,0].reshape(-1,1),t[:,1].reshape(-1,1)) 
    fig = plt.figure(figsize=(5,5),dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    l1 = ax.scatter(t[:,0], t[:,1], facecolor='k', edgecolor='k', s=80, alpha=1)
    xl = np.arange(10,50,0.1)
    ax.plot(xl, regr.predict(np.array(xl).reshape(-1,1)), color=color1[0],linewidth=4) #peru royalblue
    
    ax.set_ylim([1,121])
    ax.set_xlim([1,50])
    ax.set_xlabel('Ownership index')
    ax.set_ylabel('P(C=1|Data) index')
    ax.set_xticks([20,40])
    ax.set_yticks([40,80])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid('off')
    ax.set_facecolor('w')
    ax.xaxis.set_label_coords(0.5, -0.16)
    ax.yaxis.set_label_coords(-0.16, 0.5)
    plt.show()


#%% Pcom index arm vs wood (Fig 4D)
with plt.rc_context(params):    
    f_subhw = plt.figure(figsize=(3.5, 3.2), dpi=100, facecolor='w', edgecolor='k')
    ax = f_subhw.add_subplot(111)
#    ax.scatter(thand[:,0], twood[:,0],facecolor = 'gray', edgecolor='k',marker='o', alpha=0.9, s=20)
    ax.scatter(thand[:,1], twood[:,1],facecolor = '#333333', edgecolor='k',marker='o', alpha=0.9, s=20)
    ax.plot([0,120],[0,120],'k', linewidth=2)
    ax.set_xlim(0,120)
    ax.set_ylim(0,120)
    ax.set_xticks([0,50,100])
    ax.set_yticks([0,50,100])
    ax.set_xlabel('Arm')
    ax.set_ylabel('Wood')
    plt.show()
    
    ds = plt.figure(figsize=(3.2, 3),dpi=100, facecolor='w', edgecolor='k')
    ax = plt.subplot(1,1,1)
    ax.hist(thand[:,1]-twood[:,1], bins=16, color = '#333333', edgecolor='k', alpha=0.9)
    ax.plot(np.mean(thand[:,1]-twood[:,1]), 4.5, color='k', marker = 'v', markersize=10)
    ax.text(np.mean(thand[:,1]-twood[:,1])+2, 5,'22.9', fontsize = 20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-40,42)
    ax.set_ylim(-0,5)
    plt.tight_layout()
    plt.show()
    
#%% Pcom index arm vs wood (Fig 4D)
with plt.rc_context(params):    
    fig = plt.figure(figsize=(1.6, 1.5),dpi=100, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    ax.bar([0,0.5], np.mean([thand[:,1],twood[:,1]],1), align='center',
            yerr = stats.sem([thand[:,1],twood[:,1]],1), error_kw=dict(ecolor='black',elinewidth=1),width=0.3,edgecolor = "none",
            color=[color2[t] for t in range(2)])
    ax.set_xticks([0,0.5])
    ax.set_yticks([0,20,40])
    ax.set_xticklabels(['Arm','Wood'], fontsize=11)
    ax.set_yticklabels(['0', '20','40'], fontsize=11)
    plt.show()
    
t_pc1, p_pc1 = stats.ttest_rel(thand[:,1],twood[:,1])

#%% plot R value of ownerhip and pc1 in arm wood; Fig 2I
import seaborn as sns
simres = corr_hw(own[0],allpc1[0])

ss = plt.hist(simres[:,0],bins=7)
#    with plt.style.context(('ggplot')):
with plt.rc_context(params):
    f_hist = plt.figure(figsize=(9/2.54, 9/2.54), dpi=100)
    #    sns.distplot(simres[:,0],bins=7,kde=False,rug=False)
    ax = f_hist.add_subplot(111)
    ax.bar(ss[1][0],ss[0][0],align='edge',width=np.diff(ss[1])[0],clip_on=False, color = 'None', hatch="//", edgecolor='k', linewidth=1)
    ax.bar(ss[1][1:-1],ss[0][1:],align='edge',width=np.diff(ss[1])[0],clip_on=True, color = '#333333', edgecolor='k', linewidth=1)
    ax.set_xlim(0.45,1)
    ax.set_xlabel('R-value')
    ax.set_ylabel('Number of subjects')
    plt.show()
    
#%% plot drift; Fig supp Figure S3A; Fig 2C D
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
path = r'E:\Wen_projects\Multisensory_integration\Data_human\Block_design'
filename = 'hw_normalized_c3.mat'
data = sio.loadmat(os.path.join(path, filename))['hw']
simudata = sio.loadmat(os.path.join(path, 'causal_fit_result', 'HW_sym_meanpc_hand_p15v10_2.mat'))['Data'][0,0]
subselect = np.arange(17) # fig 2CD: sub1, sub6
with plt.rc_context(params):
    fig = plt.figure(figsize=(31/2.54, 20/2.54), dpi=100, tight_layout=True)
    for i in range(len(subselect)):
        ax = fig.add_subplot(3,6,i+1)
        subdata = pd.DataFrame(data[(data[:,0]==subselect[i]+1) & (data[:,1]==1)][:,[4,12]], columns={'rot', 'drift'})
        subdata = subdata.rename(columns={'rot':'drift', 'drift':'rot'})
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
        subsimu = pd.DataFrame(subsimu, columns={'rot', 'drift'})
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
#%% Fig 1E
color1 = ['#3A5FCD','#7EC0EE']
color2 = ['#8B0000','#EEA9B8']              
path = r'D:\Projects\MI\BCI_data\Fig1'
filename = 'hw_normalized_c3.mat'
data = sio.loadmat(os.path.join(path, filename))['hw']
simudata = sio.loadmat(os.path.join(r'D:\Projects\MI\BCI_data\Fig2', 'HW_sym_meanpc_hand_p15v10_2.mat'))['Data'][0,0]
subselect = np.arange(17)
allsubmean = np.zeros((70,17))
with plt.rc_context(params):
    fig = plt.figure(figsize=(31/2.54, 20/2.54), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    for i in range(len(subselect)):
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
        ax.plot(xi, yi, color='#41c1f0')
        allsubmean[:,i] = yi
    allmean = allsubmean.mean(1)
    allse = allsubmean.std(1)/np.sqrt(17)
    ax.plot(xi, allmean, color=color1[0])
    ax.fill_between(xi, allmean-allse, allmean+allse, color='gray',alpha=0.3)

#%% Fig 4C
color1 = ['#3A5FCD','#7EC0EE']
color2 = ['#8B0000','#EEA9B8']              
path = r'E:\Wen_projects\Multisensory_integration\SuppFig'
filename = 'hw_normalized_c3.mat'
data = sio.loadmat(os.path.join(path, filename))['hw']
simudata = sio.loadmat(os.path.join(path, 'HW_sym_meanpc_hand_p15v10_2.mat'))['Data'][0,0]
subselect = np.arange(17)
allsubmean = np.zeros((70,17))
allsubmeanw = np.zeros((70,17))
with plt.rc_context(params):
    fig = plt.figure(figsize=(4, 3), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    for i in range(len(subselect)):
        subdata = pd.DataFrame(data[(data[:,0]==subselect[i]+1) & (data[:,1]==1)][:,[4,12]], columns={'rot', 'drift'})
        # subdata = subdata.rename(columns={'rot':'drift', 'drift':'rot'})
        subdata['drift'] *= -1
        idx1 = (subdata['rot']<0) & (subdata['drift']<7) & (subdata['drift'].abs()<subdata['rot'].abs()+7)
        idx2 = (subdata['rot']>0) & (subdata['drift']>-7) & (subdata['drift'].abs()<subdata['rot'].abs()+7)
        idx3 = (subdata['rot']==0) & (subdata['drift'].abs()<7)
        idx = idx1 | idx2 | idx3
        subdata = subdata.loc[idx,:]
        submean = subdata.groupby('rot', as_index=False).mean()
        ax.scatter(submean['rot'], submean['drift'], s=2, color='k')
        model = np.polyfit(submean['rot'], submean['drift'], 3)
        xi = np.arange(-35,35)
        yi = np.polyval(model, xi)
        allsubmean[:,i] = yi
        
        subdata = pd.DataFrame(data[(data[:,0]==subselect[i]+1) & (data[:,1]==2)][:,[4,12]], columns={'rot', 'drift'})
        # subdata = subdata.rename(columns={'rot':'drift', 'drift':'rot'})
        subdata['drift'] *= -1
        idx1 = (subdata['rot']<0) & (subdata['drift']<7) & (subdata['drift'].abs()<subdata['rot'].abs()+7)
        idx2 = (subdata['rot']>0) & (subdata['drift']>-7) & (subdata['drift'].abs()<subdata['rot'].abs()+7)
        idx3 = (subdata['rot']==0) & (subdata['drift'].abs()<7)
        idx = idx1 | idx2 | idx3
        subdata = subdata.loc[idx,:]
        submean = subdata.groupby('rot', as_index=False).mean()
        ax.scatter(submean['rot'], submean['drift'], s=2, color='gray')
        model = np.polyfit(submean['rot'], submean['drift'], 3)
        xi = np.arange(-35,35)
        yi = np.polyval(model, xi)
        allsubmeanw[:,i] = yi
        
allmean = allsubmean.mean(1)
allse = allsubmean.std(1)/np.sqrt(17)
ax.plot(xi, allmean, color=color1[0])
ax.fill_between(xi, allmean-allse, allmean+allse, color='gray',alpha=0.3)

allmean = allsubmeanw.mean(1)
allse = allsubmean.std(1)/np.sqrt(17)
ax.plot(xi, allmean, color='#41c1f0')
ax.fill_between(xi, allmean-allse, allmean+allse, color='gray',alpha=0.3)



