# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  
from scipy import stats
import pandas as pd
import seaborn as sns
from scipy.stats.kde import gaussian_kde
import matplotlib as mpl



#%% Fig 2A 
mpl.rcdefaults()
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
              "grid.linewidth": 4,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'text.usetex': False,
              "xtick.color"      : "k",
              "axes.edgecolor"    :"#191919"}

# random design:
#a = sio.loadmat('E:\\Wen_projects\\Multisensory_integration\\Data_human\\Random_design\\humExp.mat')
#randomdata = pd.DataFrame(a['reach'],columns=['sub','condition','target','rotation','drift'])
#Psigma = randomdata[['sub','drift']][randomdata['condition']==2].groupby(['sub']).std()

data = sio.loadmat('HW_sym_CF_meanpc_hand_p15v10_2.mat')
simumean = data['allsimumean'][0]
datamean = data['alldatamean'][0]
subnum = len(np.unique(datamean[0][:,0]))
#alldata = data['Data']
rot = np.unique(datamean[0][:,1])
select_rot_ind = list(range(0,9,1))
rot = rot[select_rot_ind] # selected rotation
aa=int((np.size(rot)+1)/2)
subind = list(range(aa,0,-1)) + list(range(aa+2,aa*2+1,1))
hsub = int(subind[-1]/2)

h2=[[],[]]
with plt.rc_context(params):
    fig = plt.figure(figsize=(4,4),dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    color1 = ['r','#7CCD7C']    
    for i in range(2):
    #fig2 = plt.figure(figsize=(20,10),dpi=60, facecolor='w', edgecolor='k')
        datf = pd.DataFrame(datamean[i],columns=['sub','rot','drift'])
        datf = datf[datf['rot'].isin(rot)]
        tmprot = np.tile(np.unique(datf['rot']),[1000,1]).reshape(-1)
        tmpsub = np.tile(np.unique(datf['sub']),[len(tmprot),1]).T.reshape(-1)
        simuf = pd.DataFrame({'sub':tmpsub,'rot':list(tmprot)*len(np.unique(datf['sub'])),'drift':simumean[i][:,select_rot_ind].reshape(-1)})
        
    #    datf = datf[datf['rot']==3]
        datf['drift'] = -datf['drift']
        datfmean = datf.groupby(['rot','sub'],as_index=False).mean()
        means = datfmean.groupby('rot').mean()
        stes = datfmean.groupby(['rot']).std()['drift']/np.sqrt(datfmean['rot'].value_counts())
    #    datfstd = datf.groupby(['rot','sub']).std()
        ax.scatter(datf['rot'][datf['sub']!=-1],datf['drift'][datf['sub']!=-1],facecolors="None",edgecolors='#3A5FCD',marker='.',s=10,alpha=0.7)               
        idx = sum([list(range(win*9000,win*9000+100)) for win in range(subnum)],[])
        ax.scatter(simuf['rot'][simuf['sub']!=-1][idx]-1.5+i*3,
                   simuf['drift'][simuf['sub']!=-1][idx],facecolors='None' ,edgecolors=color1[i],marker='.',s=10,alpha=0.5)

        
        h1, = ax.plot(rot,datfmean[datfmean['sub']!=-1].groupby('rot').mean()['drift'],linestyle='-',color='#3A5FCD',linewidth=4)    
        h2[i], = ax.plot(rot,simuf[simuf['sub']!=-1].groupby('rot').mean()['drift'],color=color1[i],linewidth=4,linestyle='--')
    #    ax.plot(rot,np.nanmean(simumean[i][0*1000:12*1000+1000,select_rot_ind],0),color=color1[i],linewidth=4,linestyle='--')
        ax.set_facecolor('w')
        ax.grid('off')
        
    
    #plt.text(3, 28, 'Pure vision',fontsize=18,fontweight="bold")
    #plt.text(12, -10, 'Pure proprioception',fontsize=18,fontweight="bold")
    #plt.title('Monkey N',fontname = "Calibri",fontsize=30,fontweight="bold",y=1.05)
    plt.legend([h1,h2[0],h2[1]],['Data','CI','FF'],loc = 'upper right',bbox_to_anchor=(0.55, 1.02),framealpha=0)
    p1 = ax.axhline(y=0, xmin=-45, xmax=45,linewidth=5,color='k',zorder=0) 
    p2 = ax.plot([-100,100], [-100,100],linewidth=5,color='gray',zorder=0)
    #ax.set_xlabel('Disparity (deg)',fontname = "Calibri",fontsize=30,fontweight="bold") #坐标轴
    #ax.set_ylabel('Drift (deg)',fontname = "Calibri",fontsize=30,fontweight="bold")
    plt.ylim([-35,35])
    plt.xlim([-40,40])
    plt.xticks([-40,-20,0,20,40])
    plt.yticks([-40,-20,0,20,40])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_label_coords(0.5, -0.13)
    ax.yaxis.set_label_coords(-0.13, 0.5)
    for axis in ['bottom','left']:
      ax.spines[axis].set_linewidth(2)

#%% 
sns.set(color_codes=True)
model=['CI','FF']

for i in range(2):   
    fig2 = plt.figure(figsize=(7.5,2.5),dpi=300, facecolor='w', edgecolor='k')

    datf = pd.DataFrame(datamean[i],columns=['sub','rot','drift'],copy=True)
    datf = datf[datf['rot'].isin(rot)]
#    datf = datf[datf['rot']==3]
    datf['drift'] = -datf['drift']
    
    for j,r in enumerate(rot):
        if subind[j]==1:
            ax2 = plt.subplot2grid(shape=(4,hsub), loc=(1,0), rowspan=2)
        elif subind[j]<6:
            ax2 = plt.subplot2grid(shape=(4,hsub), loc=(0,subind[j]-1), rowspan=2)
        else:
            ax2 = plt.subplot2grid(shape=(4,hsub), loc=(2,subind[j]-hsub-1), rowspan=2)
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
        plt.ylim([0,0.35])
        plt.xlim([-50,50])
        plt.axvline(x=0, ymin=0, ymax = 1, linestyle='-', linewidth=2, color='k')
        plt.axvline(x=r, ymin=0, ymax = 1, linestyle='--', linewidth=2, color='gray')
        plt.title('Disparity ' + str(int(r))+'$^°$',fontsize=12,fontweight="bold")
        if subind[j]!=1:
#            ax2.set_xticklabels([])
            plt.xticks(fontname = "Calibri",fontsize=12,fontweight="bold")
            ax2.set_yticklabels([])
            ax2.set_xlabel('')
        if subind[j]==1:
            plt.xticks(fontname = "Calibri",fontsize=12,fontweight="bold")
            plt.yticks(fontname="Calibri",fontsize=12,fontweight="bold")
            ax2.set_xlabel('Drift(deg)',fontname = "Calibri",fontsize=12,fontweight="bold")
            ax2.set_ylabel('Probability',fontname = "Calibri",fontsize=12,fontweight="bold")
        if subind[j]==1:
            plt.legend(loc='upper left',bbox_to_anchor=(0.18, 3),framealpha=0,prop={'size':12,'weight':'bold'})
#        sns.kdeplot(simumean[i][:,j],kernel='biw',  bw=5, shade=1,ax=ax2)
        
#    plt.close(fig2) 
plt.show()
