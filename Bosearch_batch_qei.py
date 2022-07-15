# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 15:13:57 2021

(1) This python script examplifies how to conduct parallel BO(qEI) search in COFs for CH4 storage;

(2) Initialization by Curated COFs;

(3) Parallel acquistion in Berkeley COFs;

@author: Hongjian Tang(NUS/SEU)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from scipy.stats import spearmanr,reciprocal
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.learning import RandomForestRegressor
import warnings
from scipy.stats import norm
import matplotlib.animation as animation
import qei
import pickle
import time
import random

Selected_Features =  pd.read_csv('Feature_selection.csv')
Features = Selected_Features['COF_Feature']

#1.1 Curated_COF
data_curated = pd.read_excel('COF_Curated.xlsx')
data_curated_atom = pd.read_csv('Atom_Curated.csv')
data_curated_atom.drop(columns='Unnamed: 0',inplace=True)
data_curated_bond = pd.read_csv('Bond_Curated.csv')
data_curated_bond.drop(columns='Unnamed: 0',inplace=True)
Index_valid = np.array(data_curated_atom.loc[(data_curated_atom!=0).any(axis=1),:].index)
Index_drop = np.array(data_curated.loc[data_curated['VSA [m2/cm3, Zeo++, 1.82]']==0].index)
Index_valid = pd.Int64Index(np.delete(Index_valid,[np.where(Index_valid == index) for index in Index_drop]))

Atom_curated = data_curated_atom.loc[Index_valid]
Bond_curated = data_curated_bond.loc[Index_valid]
Y_curated = data_curated.loc[Index_valid]

lcd_curated = Y_curated.iloc[:,3]
pld_curated = Y_curated.iloc[:,4]
lfpd_curated = Y_curated.iloc[:,5]
vf_curated = Y_curated.iloc[:,6]
pv_curated = Y_curated.iloc[:,7]
povf_curated = Y_curated.iloc[:,8]
ponvf_curated = Y_curated.iloc[:,9]
popv_curated = Y_curated.iloc[:,10]
ponpv_curated = Y_curated.iloc[:,11]
vsa_curated  = Y_curated.iloc[:,12]
gsa_curated  = Y_curated.iloc[:,13]
nvsa_curated  = Y_curated.iloc[:,14]
ngsa_curated  = Y_curated.iloc[:,15]
den_curated  = Y_curated.iloc[:,17]

Geo_curated =  pd.concat([lcd_curated,pld_curated,lfpd_curated,
                          vf_curated,pv_curated,povf_curated,popv_curated,
                          gsa_curated,vsa_curated,den_curated], axis=1)

CR_curated = Atom_curated.iloc[:,9]

X_curated = pd.concat([Geo_curated,Atom_curated,Bond_curated], axis=1)[Features]

#1.2 Berkeley_COF
data_berkeley = pd.read_excel('COF_Berkeley.xlsx')
data_berkeley_atom = pd.read_csv('Atom_Berkeley.csv')
data_berkeley_atom.drop(columns='Unnamed: 0',inplace = True)
data_berkeley_bond = pd.read_csv('Bond_Berkeley.csv')
data_berkeley_bond.drop(columns='Unnamed: 0',inplace = True)
Index_valid = data_berkeley_atom.loc[(data_berkeley_atom!=0).any(axis=1),:].index

Atom_berkeley = data_berkeley_atom.loc[Index_valid]
Bond_berkeley = data_berkeley_bond.loc[Index_valid]
Y_berkeley = data_berkeley.loc[Index_valid]
Geo_berkeley  = Y_berkeley[Geo_curated.columns]

X_berkeley = pd.concat([Geo_berkeley,Atom_berkeley,Bond_berkeley], axis=1)[Features]

##### Data processing #####
Name_berkeley = Y_berkeley[' name']
X_all = pd.concat([X_curated,X_berkeley], axis=0)
Y_all = pd.concat([Y_curated.iloc[:,21],Y_berkeley.iloc[:,21]], axis=0)
all_inds = set(range(len(Y_all)))
X_all.index =  all_inds
Y_all.index =  all_inds
VSA_all = X_all.iloc[:,8]
scaler = MinMaxScaler()
X_all = scaler.fit_transform(X_all)

##### Initialization for traing set #####
in_train = np.zeros(len(Y_all), dtype=bool)
in_train[:len(X_curated)] = True
train_num = in_train.sum()
print('Picked {} training entries'.format(in_train.sum()))
assert not np.isclose(np.max(Y_all), np.max(Y_all[in_train])) #Flag if random choice on the Y_max

plt.plot(VSA_all, Y_all, marker='o', alpha=0.3, 
         color='silver', linestyle='None', markersize=5, label='Full')
plt.plot(VSA_all[in_train], Y_all[in_train], marker='o', alpha=1.0, 
         color='k', linestyle='None', markersize=5, label='Full')

model = RandomForestRegressor(n_estimators=100,n_jobs=-1)
KB = qei.QEI(model = model,virtual="KB")
KBUB = qei.QEI(model = model,virtual="KBUB")
KBLB = qei.QEI(model = model,virtual="KBLB")
KBR = qei.QEI(model = model,virtual="KBR")

n_batch = 10
n_steps = int(300/n_batch)

KB_train = [list(set(np.where(in_train)[0].tolist()))]
KBUB_train = [list(set(np.where(in_train)[0].tolist()))]
KBLB_train = [list(set(np.where(in_train)[0].tolist()))]
KBR_train = [list(set(np.where(in_train)[0].tolist()))]
KB_train_inds = []
KBUB_train_inds = []
KBLB_train_inds = []
KBR_train_inds = []

start_time = time.time()

for i in range(n_steps):
    print('Search status:'); print(i)
    # KB method
    KB_train_inds = KB_train[-1].copy()  # Last iteration
    KB_search_inds = list(all_inds.difference(KB_train_inds))
    model.fit(X_all[KB_train_inds], Y_all[KB_train_inds])
    KB_index = KB.get_index(X_data = X_all,y_data = Y_all,
                            search_inds_ini = KB_search_inds,train_inds_ini = KB_train_inds,batch_size=n_batch)
    KB_train_inds += KB_index
    KB_train.append(KB_train_inds)

    # KBUB serach
    KBUB_train_inds = KBUB_train[-1].copy()  # Last iteration
    KBUB_search_inds = list(all_inds.difference(KBUB_train_inds))
    model.fit(X_all[KBUB_train_inds], Y_all[KBUB_train_inds])
    KBUB_index = KBUB.get_index(X_data = X_all,y_data = Y_all,
                                search_inds_ini = KBUB_search_inds,train_inds_ini = KBUB_train_inds,batch_size=n_batch)
    KBUB_train_inds += KBUB_index
    KBUB_train.append(KBUB_train_inds)

    # KBLB serach
    KBLB_train_inds = KBLB_train[-1].copy()  # Last iteration
    KBLB_search_inds = list(all_inds.difference(KBLB_train_inds))
    model.fit(X_all[KBLB_train_inds], Y_all[KBLB_train_inds])
    KBLB_index = KBLB.get_index(X_data = X_all,y_data = Y_all,
                              search_inds_ini = KBLB_search_inds,train_inds_ini = KBLB_train_inds,batch_size=n_batch)
    KBLB_train_inds += KBLB_index
    KBLB_train.append(KBLB_train_inds)
    
    # KBR serach
    KBR_train_inds = KBR_train[-1].copy()  # Last iteration
    KBR_search_inds = list(all_inds.difference(KBR_train_inds))
    model.fit(X_all[KBR_train_inds], Y_all[KBR_train_inds])
    KBR_index = KBR.get_index(X_data = X_all,y_data = Y_all,
                              search_inds_ini = KBR_search_inds,train_inds_ini = KBR_train_inds,batch_size=n_batch)
    KBR_train_inds += KBR_index
    KBR_train.append(KBR_train_inds)

run_time = time.time() - start_time
print('running time:'); print(run_time)

##### Save files #####
Save_path = 'Save_Bosearch_RF_QEI'
if not os.path.exists(Save_path):
    os.makedirs(Save_path)
for lists in ['KB_train_inds', 'KB_train', 
              'KBUB_train_inds', 'KBUB_train',
              'KBLB_train_inds','KBLB_train',
              'KBR_train_inds','KBR_train',
              ]:
    file_path = os.path.join(Save_path, lists+'.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(eval(lists), file)
