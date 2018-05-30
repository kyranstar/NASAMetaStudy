# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:58:36 2018

@author: Kyran Adams
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
import scipy.stats as stats

def plot_lasso(data, alphas=np.arange(0.01, 1.0, 0.01)):
    mat = np.zeros((data.shape[1]-1, len(alphas)))
    for i, alpha in enumerate(alphas):
        model = Lasso(alpha=alpha)
        trainx = data.drop(['y'], axis=1)
        trainy = data[['y']]
        model.fit(trainx, trainy)
        mat[:, i] = model.coef_
    # For each coefficient
    for i in range(mat.shape[0]):
        plt.plot(alphas, mat[i, :])
    plt.legend(range(1, mat.shape[0]+1))
    plt.show()

def plot_normal(h):
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))
    plt.plot(h,fit,'-o')    
    plt.hist(h,normed=True)      #use this to draw histogram of your data    
    plt.show()                   #use may also need add this 