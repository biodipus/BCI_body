# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  
from scipy import stats
import pandas as pd
import seaborn as sns
from scipy.stats.kde import gaussian_kde
import matplotlib as mpl
#import matplotlib.style


#%%#mpl.rcdefaults()
params = {"figure.facecolor": "w",
              'figure.autolayout' : True,
              'font.family': 'Calibri',
              'font.weight': 'bold',
              'font.size': 22.0,
              'axes.titlesize' : 22,
              'axes.titleweight': 'bold',
              'axes.labelsize': 22,
              'axes.labelweight': 'bold',
              'legend.fontsize': 22,
              "axes.facecolor": "w",
              'figure.subplot.wspace': 0.3,
              "axes.grid" : False,
              "axes.grid.axis" : "y",
              "grid.color"    : "#ffffff",
              "grid.linewidth": 4,
              'xtick.labelsize': 22,
              'ytick.labelsize': 22,
              'text.usetex': False,
              "xtick.color"      : "#191919",
              "axes.edgecolor"    :"#191919"}
#mpl.rcParams.update(params)

#%%
def reset_matplotlib():
    """
    Reset matplotlib to a common default.
    """
    # Set all default values.
    mpl.rcdefaults()
    # Force agg backend.
    plt.switch_backend('agg')
    # These settings must be hardcoded for running the comparision tests and
    # are not necessarily the default values.
    mpl.rcParams['font.family'] = 'Bitstream Vera Sans'
    mpl.rcParams['text.hinting'] = False
    # Not available for all matplotlib versions.
    try:
        mpl.rcParams['text.hinting_factor'] = 8
    except KeyError:
        pass
#%% Fig 1F
data = sio.loadmat('N_HW_sym_Norc3_4day_CF.mat')
simumean = data['allsimumean'][0]
datamean = data['alldatamean'][0]
subnum = len(np.unique(datamean[0][:,0]))

#alldata = data['Data']
rot = np.unique(datamean[0][:,1])
select_rot_ind = list(range(0,len(rot),1))
rot = rot[select_rot_ind] # selected rotation
aa=int((np.size(rot)+1)/2)
subind = list(range(aa,0,-1)) + list(range(aa+2,aa*2+1,1))
hsub = int(subind[-1]/2)

with plt.rc_context(params):
    fig = plt.figure(figsize=(4,4),dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    color1 = ['r','#7CCD7C']
    for i in range(2):  
        datf = pd.DataFrame(datamean[i],columns=['sub','rot','drift'],copy=True)
        datf = datf[datf['rot'].isin(rot)]
        idx1 = (datf['rot']<0) & (datf['drift']<7) & (datf['drift'].abs()<datf['rot'].abs()+7)
        idx2 = (datf['rot']>0) & (datf['drift']>-7) & (datf['drift'].abs()<datf['rot'].abs()+7)
        idx3 = (datf['rot']==0) & (datf['drift'].abs()<7)
        idx = idx1 | idx2 | idx3
        datf = datf.loc[idx,:]
        tmprot = np.tile(np.unique(datf['rot']),[1000,1]).reshape(-1)
        tmpsub = np.tile(np.unique(datf['sub']),[len(tmprot),1]).T.reshape(-1)
        
        datf['drift'] = datf['drift']
        datfmean = datf.groupby(['rot','sub'],as_index=False).mean()
        means = datfmean.groupby('rot').mean()
        stes = datfmean.groupby(['rot']).std()['drift']/np.sqrt(datfmean['rot'].value_counts())
        ax.scatter(datf['rot'][datf['sub']!=-1],datf['drift'][datf['sub']!=-1],facecolors='None',edgecolors='k',marker='.',s=10,alpha=0.7)               
        
        dd = datf[datf['sub']!=-1]
        ddmean = dd.groupby('rot', as_index=False).mean()
        model = np.polyfit(ddmean['rot'], ddmean['drift'], 3)
        xi = np.arange(-45,45)
        yi = np.polyval(model, xi)
        ax.plot(xi, yi, color='#3A5FCD')
        ax.set_facecolor('w')
        ax.grid('off')
        
    p1 = ax.axhline(y=0, xmin=-45, xmax=45,linewidth=5,color='k',zorder=0)    
    p2 = ax.plot([-100,100], [-100,100],linewidth=5,color='gray',zorder=0)
    plt.ylim([-45,45])
    plt.xlim([-50,50])
    plt.xticks([-40,-20,0,20,40])
    plt.yticks([-40,-20,0,20,40])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_label_coords(0.5, -0.13)
    ax.yaxis.set_label_coords(-0.13, 0.5)
    for axis in ['bottom','left']:
      ax.spines[axis].set_linewidth(2)
    plt.show()
#%%
data = sio.loadmat('N_HW_sym_Norc3_4day_CF.mat')
simumean = data['allsimumean'][0]
datamean = data['alldatamean'][0]
alldata = data['Data']
subnum = len(np.unique(datamean[0][:,0]))
rot = np.unique(datamean[0][:,1])
select_rot_ind = list(range(0,9,1))
rot = rot[select_rot_ind] # selected rotation
aa=int((np.size(rot)+1)/2)
subind = list(range(aa,0,-1)) + list(range(aa+2,aa*2+1,1))
#subind = [7,6,5,4,3,2,1,9,10,11,12,13,14]
#subind = [5,4,3,2,1,7,8,9,10]
hsub = int(subind[-1]/2)
          
day = 1

with plt.rc_context(params):
    fig = plt.figure(figsize=(6,6),dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    color1 = ['r','#7CCD7C','#1f78b4']    
    for i in range(2):    
        datf = pd.DataFrame(datamean[i],columns=['sub','rot','drift'],copy=True)
        datf = datf[datf['rot'].isin(rot)]
        tmprot = np.tile(np.unique(datf['rot']),[1000,1]).reshape(-1)
        tmpsub = np.tile(np.unique(datf['sub']),[len(tmprot),1]).T.reshape(-1)
        simuf = pd.DataFrame({'sub':tmpsub,'rot':list(tmprot)*len(np.unique(datf['sub'])),'drift':simumean[i][:,select_rot_ind].reshape(-1)})
        
    #    datf = datf[datf['rot']==3]
        datfmean = datf.groupby(['rot','sub'],as_index=False).mean()
        means = datfmean.groupby('rot').mean()
        stes = datfmean.groupby(['rot']).std()['drift']/np.sqrt(datfmean['rot'].value_counts())
        ax.scatter(datf['rot'][datf['sub']!=-1],datf['drift'][datf['sub']!=-1],facecolors='None',edgecolors='#3A5FCD',marker='.',s=50,alpha=1)               
        idx = sum([list(range(win*9000,win*9000+500)) for win in range(subnum)],[])
        ax.scatter(simuf['rot'][simuf['sub']!=-1][idx]-1.5+i*3,
                   simuf['drift'][simuf['sub']!=-1][idx],facecolors='None',edgecolors=color1[i],marker='.',s=50,alpha=1)
        
        ax.plot(rot,datfmean[datfmean['sub']!=-1].groupby('rot').mean()['drift'],linestyle='-',color='#3A5FCD',linewidth=5)    
        ax.plot(rot,simuf[simuf['sub']!=-1].groupby('rot').mean()['drift'],color=color1[i],linewidth=5,linestyle='--')
        
    p1 = ax.axhline(y=0, xmin=-45, xmax=45,linewidth=5,color='k',zorder=0)    
    p2 = ax.plot([-100,100], [-100,100],linewidth=5,color='gray',zorder=0)
    ax.set_xlabel('Disparity (deg)') #坐标轴
    ax.set_ylabel('Drift (deg)')
    ax.set_facecolor('w')
    ax.grid('off')
    plt.ylim([-51,51])
    plt.xlim([-50,50])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_label_coords(0.5, -0.13)
    ax.yaxis.set_label_coords(-0.13, 0.5)
    for axis in ['bottom','left']:
      ax.spines[axis].set_linewidth(2)
      
    plt.show()
#%% Figure S2
model=['CI','FF']
sns.set(color_codes=True)

with plt.rc_context(params):    
    for i in range(0,2):
        datf = pd.DataFrame(datamean[i],columns=['sub','rot','drift'],copy=True)
    #    datf = datf[datf['sub']==3]
    #    datf['drift'] = -datf['drift']
        fig2 = plt.figure(figsize=(12.5,4),dpi=80, facecolor='w', edgecolor='k')
        
        for j,r in enumerate(rot):
            if subind[j]==1:
                ax2 = plt.subplot2grid(shape=(4,hsub), loc=(1,0), rowspan=2)
            elif subind[j]<6:
                ax2 = plt.subplot2grid(shape=(4,hsub), loc=(0,subind[j]-1), rowspan=2)
            else:
                ax2 = plt.subplot2grid(shape=(4,hsub), loc=(2,subind[j]-hsub-1), rowspan=2)
    #        ax2 = fig2.add_subplot(2,5,subind[j])
            plt.tight_layout()
            x = datf['drift'][datf['rot']==r]
            x = x[~np.isnan(x)]
            
    #        kde = gaussian_kde(x)
    #        # these are the values over wich your kernel will be evaluated
    #        dist_space = np.linspace(np.min(x)-10, np.max(x)+10, 100 )
    #        # plot the results
    #        ax2.plot(dist_space, kde(dist_space))
    #        
    #        x = simumean[i][:,j]
    #        x = x[~np.isnan(x)]
    #        kde = gaussian_kde(x)
    #        # these are the values over wich your kernel will be evaluated
    #        dist_space = np.linspace(np.min(x)-10, np.max(x)+10, 100 )
    #        # plot the results
    #        ax2.plot(dist_space, kde(dist_space))
            h1 = sns.distplot(x,hist=1,kde=1,norm_hist=1,label='Data')
    #        sns.distplot(datf['drift'][datf['rot']==r],kernel='biw', bw=5, shade=1, ax=ax2)
            x = simumean[i][:,select_rot_ind[j]]
            x = x[~np.isnan(x)]
    #        x2 = simumean[1][:,select_rot_ind[j]]
    #        x2 = x2[~np.isnan(x2)]
            h2 = sns.distplot(x,hist=1,kde=1,norm_hist=1,color=color1[i],label=model[i])
    #        h2 = sns.distplot(x2,hist=1,kde=1,norm_hist=1,color=color1[1],label='FF')
            
    #        Vx = np.random.normal(r,1, size=1000)
    #        sns.distplot(Vx,color="gray");        
            plt.ylim([0,0.25])
            plt.xlim([-50,50])
            plt.axvline(x=0, ymin=0, ymax = 1, linestyle='-', linewidth=2, color='k')
            plt.axvline(x=r, ymin=0, ymax = 1, linestyle='--', linewidth=2, color='gray')
            plt.title('Disparity ' + str(int(r)) + '$^°$')
            if subind[j]!=1:
    #            ax2.set_xticklabels([])
#                plt.xticks(fontname = "Calibri",fontsize=12,fontweight="bold")
                ax2.set_yticklabels([])
                ax2.set_xlabel('')
            if subind[j]==1:
#                plt.xticks(fontname = "Calibri",fontsize=12,fontweight="bold")
#                plt.yticks(fontname="Calibri",fontsize=12,fontweight="bold")
                ax2.set_xlabel('Drift(deg)')
                ax2.set_ylabel('Probability')
            if subind[j]==1:
                plt.legend(loc='upper left',bbox_to_anchor=(-0.25, 3))
                
    #    plt.close(fig2) 
    plt.show()
    