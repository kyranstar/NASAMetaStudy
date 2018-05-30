# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:58:26 2018

@author: Kyran Adams
"""
import numpy as np
import pandas as pd
import math

def get_distribution_samples(data, num_samples, true_model, y_variance=1.0):
    """
    Given a set of normally distributed data, creates num_samples points of
    simulated data.
    """
    # Generate xs as multivariate normal variables
    xs = data.drop(['y'], axis=1)
    mean = xs.mean(axis=0)
    cov = xs.corr()
    samples = np.random.multivariate_normal(mean, cov, size=num_samples)
    
    df = pd.DataFrame(data=samples, columns=xs.columns)
    
    # Calculate the error term in the regression
    #if not y_variance:
    #    sse = np.square(data[['y']].values - true_model.predict(xs)).sum()
    #    y_variance = sse/(data.shape[0] - data.shape[1])#np.array(data[['y']].values - true_model.predict(xs)).var()
    #    print(y_variance)
    df.loc[:, 'y'] = true_model.predict(samples) + np.random.normal(0, math.sqrt(y_variance), num_samples)
    return df