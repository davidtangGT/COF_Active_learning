# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:46:31 2022

qEI function used in parallel BO search

@author: Hongjian Tang(NUS/SEU)
"""

import numpy as np
from scipy.stats import norm
import copy

class QEI(object):
    def __init__(self,model,virtual):
        """
            @model              sklearn model can return std.
            @virtual            The method to get qEI
                                options=["KB", "KBLB", "KBUB", "KBR"]
        """
        self.model = model
        self.virtual = virtual
        
    def get_index(self, X_data,y_data,search_inds_ini,train_inds_ini,batch_size=10):
        search_inds = copy.deepcopy(search_inds_ini)
        train_inds = copy.deepcopy(train_inds_ini)
        batch_size = batch_size
        index_selection = []
        y_k = np.array([])
        y_train = y_data[train_inds]
        # Virtual enrichement loop
        for i in range(batch_size):
            X_train = X_data[train_inds]
            X_serach = X_data[search_inds]
            y_train = np.atleast_2d(np.append(y_train, y_k))
            self.model.fit(X_train,y_train.T)	
            # xt best x-coord point to evaluate
            EI_value = self.EI(
                   X = X_serach, y_opt = np.max(y_train), xi=0.01,
                   )
            # Set temporaly the y-coord point based on the kriging prediction
            x_k_index = search_inds[np.argmax(EI_value)]
            x_k = X_data[search_inds[np.argmax(x_k_index)]]
            y_k = self.get_virtual_point(x_k,y_train)

            # Update y_data with predicted value
            index_selection += [x_k_index]
            train_inds += [x_k_index]
            search_inds = list(set(search_inds).difference(index_selection))
            # print("pick virtual point"); print(i)
        return index_selection

    def EI(self, X, y_opt, xi=0.01):
        """Expected improvement"""
        mu, std = self.model.predict(X, return_std=True)
        # check dimensionality of mu, std so we can divide them below
        if (mu.ndim != 1) or (std.ndim != 1):
            raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                             "however both must be 1-dimensional. Did you train "
                             "your model with an (N, 1) vector instead of an "
                             "(N,) vector?"
                             .format(mu.ndim, std.ndim))
        ei = np.zeros_like(mu)
        mask = std > 0
        improve = mu[mask] - y_opt - xi 
        scaled = improve / std[mask]
        cdf = norm.cdf(scaled)
        pdf = norm.pdf(scaled)
        exploit = improve * cdf
        explore = std[mask] * pdf
        ei[mask] = exploit + explore
        return ei

    def get_virtual_point(self, x, y_data):
    
        x= np.array(x, ndmin = 2)
        virtual = self.virtual

        if virtual == "KB":
            return self.model.predict(x)
        
        if virtual == "KBUB":
            conf = 3

        if virtual == "KBLB":
            conf = -3

        if virtual == "KBR":
            conf = np.random.randn()
            
        pred,std  = self.model.predict(x,return_std=True)

        return pred + conf * np.sqrt(std)