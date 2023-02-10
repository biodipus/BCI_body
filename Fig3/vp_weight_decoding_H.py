# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.io as sio
import os
from sklearn.utils import resample
import random
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVR  
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt
from scipy import stats


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

#### permutation test
def permu_stats(permu_res, predict_res):
    
    predict_res = predict_res.mean()
    permu_res = permu_res.reshape((-1,3)).mean(1)
    
    pvalue = (np.sum(permu_res >= predict_res) + 1.0) / (len(permu_res) + 1)
    
    return pvalue

###########%
path = r'***\Fig3' # give the full path
vpweight_filename = 'HallWeightHoldThresh3trls.mat'

prob = sio.loadmat(os.path.join(path, vpweight_filename))
vpweight = prob['prob'][:,:,:,0]
conds = prob['conds']
trial = 100

subselect = sio.loadmat(os.path.join(path, 'VPTselID_H_HW.mat'))['id3'][:,0]
subselect = np.where(subselect == 1)[0]
vpweight = vpweight[0:100, 2:, subselect]
conds = conds[conds[:,0]==3]

simu = sio.loadmat(os.path.join(path, 'H_HW_sym_Norc3_meanpc.mat'))
drift = simu['Data'][0,0]
pc1 = simu['allpc1'][0][0]
edges = [-100,-35,-25,-15,-6,6,15,25,35,100]
rotation = np.unique(conds[:,1]).astype(int)
pc1_all = np.zeros((trial, conds.shape[0], len(subselect)))
pc1_all[:] = np.NaN

s = 0
for i in subselect:
    pc1_sub = pc1[0][i]
    dft_sub = drift[0,i][0,1]   
    c = 0
    for j in range(len(rotation)):
        sort_ind = np.argsort(dft_sub[:, j], 0)
        dft_rot = dft_sub[sort_ind, j]
        pc1_rot = pc1_sub[sort_ind, j]
        pc1_drift = pd.DataFrame({'pc1':pc1_rot, 'drift': dft_rot})
        if rotation[j] < 0:
            pc1_drift = pc1_drift[pc1_drift['drift'] < 7]
            hist, hist_edge = np.histogram(dft_rot,edges)
            pc1_drift['group'] = pd.cut(pc1_drift['drift'], hist_edge, labels=False)            
            for n in conds[conds[:,1]==rotation[j],2]:
                nn = n - 1
                tmp = pc1_drift['pc1'][pc1_drift['group']==nn]
                if tmp.size > 1:
                    pc1_all[:,c,s] = np.nanmean(np.random.choice(tmp, (tmp.size, trial)), axis=0)
                c += 1
        if rotation[j] >= 0:
            pc1_drift = pc1_drift[pc1_drift['drift'] > -7]
            hist, hist_edge = np.histogram(dft_rot,edges)
            pc1_drift['group'] = pd.cut(pc1_drift['drift'], hist_edge, labels=False)
            for n in conds[conds[:,1]==rotation[j],2]:
                nn = n - 1
                tmp = pc1_drift['pc1'][pc1_drift['group']==nn]
                if tmp.size > 1:
                    pc1_all[:,c,s] = np.nanmean(np.random.choice(tmp, (tmp.size, trial)), axis=0)
                c += 1
               
              
    s += 1

sd = np.nanmean(np.nanstd(vpweight,0),0)
vpweight_sd = vpweight[:, :, :] 
neu_num = vpweight_sd.shape[2]
pcc = np.nanmean(pc1_all,2)
for i in range(29):
    pcc[:,i] = np.nanmean(pcc[:,i])
pc1_all_flat = pcc.reshape(-1)
vpweight_flat = vpweight_sd.reshape(-1,neu_num)

#%% SVM
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer


#### kmean cluster ####    
class_c = np.tile(conds[:,2], trial) - 1
class_i = np.tile(np.arange(29), trial)
pc1_all_df = pd.DataFrame({'pc1_all': pc1_all_flat, 'class': class_i})

n_clusters = 8

clusterer  = KMeans(n_clusters=n_clusters) 
pc1_all_df['label'] = clusterer.fit_predict(pc1_all_flat.reshape(-1,1))
c_ind = pc1_all_df.groupby('label').mean().sort_values('pc1_all').index.values
pc1_all_df['label'].replace(c_ind, np.arange(n_clusters),  inplace=True)

#### svc params ####
n_fold = 3
n_repeats = 10
rkf = RepeatedKFold(n_splits=n_fold, n_repeats=n_repeats) 

train_ind = []
test_ind = []
scores = []
balanced_acc = []
confu_mat = np.zeros((n_clusters, n_clusters, n_fold*n_repeats))
sample_num = len(pc1_all_flat)
predict_acc = np.zeros((n_clusters, n_clusters, n_fold*n_repeats))

#### SVC decoding ####
k = 0
for train, test in rkf.split(np.arange(trial)):
    train_ind.append(train)
    test_ind.append(test)
    
    x_train = vpweight_sd[train, :]
    for n in range(x_train.shape[2]):
        for j in range(x_train.shape[1]):
            tmp = x_train[:,j,n]
            x_train[:,j,n] = np.nanmean(np.random.choice(tmp, (tmp.size, tmp.size)), axis=0)
    
    x_train = x_train.reshape(-1,neu_num)   
    empty_neuron = np.isnan(x_train).all(0)
    x_train[:, empty_neuron] = np.nanmean(x_train)#, 1).reshape(-1,1)
    y_train = pc1_all_df.loc[:,'label'].values.reshape(-1,29)[train,:].reshape(-1)
    
    x_test = vpweight_sd[test, :]
    for n in range(x_test.shape[2]):
        for j in range(x_test.shape[1]):
            tmp = x_test[:,j,n]
            x_test[:,j,n] = np.nanmean(np.random.choice(tmp, (tmp.size, tmp.size)), axis=0)
    x_test = x_test.reshape(-1,neu_num) 
    empty_neuron = np.isnan(x_test).all(0)
    x_test[:, empty_neuron] = np.nanmean(x_test)#, 1).reshape(-1,1)
    y_test = pc1_all_df.loc[:,'label'].values.reshape(-1,29)[test,:].reshape(-1)
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x_train)
    x_train = imp.transform(x_train)
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x_test)
    x_test = imp.transform(x_test)
    
    clf = SVC(kernel='linear', C=1, class_weight= {3:0.6, 5:0.4, 6:0.15, 7:0.12}, gamma='scale') # delay best:{1:0.7,3:0.5, 5:0.6, 6:0.15, 7:0.12} {1:0.6, 3:0.4, 6:0.22, 7:0.12}
    clf.fit(x_train, y_train)
    scores.append(clf.score(x_test, y_test))
    
    y_hat = clf.predict(x_test)
    balanced_acc.append(balanced_accuracy_score(y_test, y_hat))
    confu_mat[:,:,k] = confusion_matrix(y_test, y_hat)
    
    result = pd.DataFrame({'y_test': y_test, 'y_hat': y_hat})

    k += 1 
    print([k,np.mean(balanced_acc[-3:])])
    
#predict_acc_mean = predict_acc.mean(2)
print(np.mean(scores))
print(np.mean(balanced_acc))
### plot accuracy ####
confu_mat_mean = confu_mat.mean(2)
confu_mat_mean = confu_mat_mean.astype('float')/confu_mat_mean.sum(axis=1)[:, np.newaxis]
with plt.rc_context(params):
    fig = plt.figure(figsize = (4, 3.5), dpi = 80, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    h1 = ax.pcolormesh(confu_mat_mean, cmap=plt.cm.Reds, vmin=0, vmax=1) #RdBu Reds
    ax.invert_yaxis()
    ax.set_xticks([0,2,4,6,8])
    ax.set_xticklabels([0,2,4,6,8])
    ax.set_yticks([0,2,4,6,8])
    ax.set_yticklabels([0,2,4,6,8])
    ax.set_xlabel('Predicted Pcom')
    ax.set_ylabel('Actual Pcom')
    fig.colorbar(h1)

    





    