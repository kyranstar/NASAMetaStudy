# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:30:08 2018

@author: Kyran Adams
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import train_test_split

def select_variables(model, data):
    """
    Given a list of regression coefficients, gives a set of the 1-based indices of 
    non-zero coefficients. Example: [0.5, 0.4, 0.0, 3.0] -> {1, 2, 4}
    """
    all_feats = data.drop(["y"], axis=1)
    model.fit(all_feats, data[["y"]])
    coef = list(np.array(model.coef_).flat)
    return set([i+1 for i,x in enumerate(coef) if x != 0.0])



def get_distribution_samples(data, num_samples):
    """
    Given a set of normally distributed data, creates num_samples points of
    simulated data.
    """
    mean = data.mean(axis=0)
    cov = data.corr()
    samples = np.random.multivariate_normal(mean, cov, size=num_samples)
    return pd.DataFrame(data=samples, columns=data.columns)
  
    

def find_best_alpha_mse(data):
    """
    Attempts to find the best lasso alpha value using generalized prediction mse.
    """
    alphas = np.arange(0.1, 1.0, 0.05)
    def mse(alpha):
        #model.fit(predictors, data[['y']])
        score = 0.0
        cv = 10
        for i in range(cv):
            # create and train model in cv
            model = Lasso(alpha=alpha)
            train, test = train_test_split(data, test_size=1.0/cv)
        
            trainx = train.drop(["y"], axis=1)
            trainy = train[['y']]
            model.fit(trainx, trainy)
            
            testx = test.drop(["y"], axis=1)
            testy = test[['y']]
            score += mean_squared_error(testy, model.predict(testx))
        
        #score = -cross_val_score(model, predictors, data[["y"]], cv=10, 
        #                         scoring='neg_mean_squared_error').mean()
        return score/cv
    scores = np.array([mse(a) for a in alphas])
    stddev = scores.std()
    minscore = scores.min()
    # Remove all alphas whose mse is above one standard deviation from min mse  
    alpha_score = filter(lambda t: t[1] <= minscore + stddev, zip(alphas, scores))

    best_alpha = np.array(list(map(lambda a: a[0], alpha_score))).max()

    return best_alpha

def find_best_alpha_bic(data):
    """
    Attempts to find the best lasso alpha value fitting a dataset using the BIC
    metric: https://stats.stackexchange.com/questions/126898/tuning-alpha-parameter-in-lasso-linear-model-in-scikitlearn
    """
    predictors = data.drop(["y"], axis=1)
    alphas = np.arange(0.1, 1.0, 0.05)
    best_score = np.inf
    best_alpha = 0
    def bic_score(predictors, y, model):
        sse = sum((model.predict(predictors) - y.values[0])**2)
        s = np.count_nonzero(model.coef_)   
        n = len(predictors.columns)
        cn = math.sqrt(n)/(s*s)
        print(math.log(sse/n) + s*math.log(n)/n*cn)
        return math.log(sse/n) + abs(s)*math.log(n)/n*cn
        
    for alpha in alphas:
        model = Lasso(alpha=alpha)
        model.fit(predictors, data[['y']])
        score = bic_score(predictors, data[['y']], model)
        if score < best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha


def plot_subset_accuracy(df, sample_range, trials):
    """
    Plots various metrics of how accurately lasso can select the true variables
    given certain numbers of samples.
    Arguments:
        df: The dataset
        sample_range: A list of numbers of samples to try (x-axis values)
        trials: The number of trials to do for each sample size
    """
    best_t = find_best_alpha_mse(df)
    print(best_t)
    #print(select_variables(LinearRegression(), df))
    true_model = Lasso(alpha=best_t)
    true_variables = select_variables(true_model, df)
    print(true_variables)

    
    arr_perfectly_chosen = []
    arr_predictors_missed = []
    arr_false_predictors_chosen = []
    arr_symm_diff = []
    arr_symm_diff_2 = []
    
    for num_samples in sample_range:
        
        num_perfectly_chosen = 0.0
        num_predictors_missed = 0.0
        num_false_predictors_chosen = 0.0
        num_symm_diff = 0.0
        num_symm_diff_2 = 0.0
        for t in range(trials):    
            sampled_data = get_distribution_samples(df, num_samples)
            best_alpha = find_best_alpha_mse(sampled_data)
            
            model = Lasso(alpha=best_alpha)
            chosen_variables = select_variables(model, sampled_data)
            print(best_alpha)
            print(len(chosen_variables))
            if chosen_variables == true_variables:
                num_perfectly_chosen += 1
            if len(chosen_variables) - len(true_variables.intersection(chosen_variables)) <= 1:
                num_predictors_missed += 1
            if len(true_variables) - len(true_variables.intersection(chosen_variables)) <= 1:
                num_false_predictors_chosen += 1
            if len(true_variables.symmetric_difference(chosen_variables)) <= 1:
                num_symm_diff += 1
            if len(true_variables.symmetric_difference(chosen_variables)) <= 2:
                num_symm_diff_2 += 1
        
        print("With %d samples:" % num_samples)
        print("%0.2f%% trials perfectly chosen" % (num_perfectly_chosen/trials*100))
        arr_perfectly_chosen.append(num_perfectly_chosen/trials*100)
        print("%0.2f%% trials missed <= 1 predictors" % (num_predictors_missed/trials*100))
        arr_predictors_missed.append(num_predictors_missed/trials*100)
        print("%0.2f%% trials had <= 1 false predictors" % (num_false_predictors_chosen/trials*100))
        arr_false_predictors_chosen.append(num_false_predictors_chosen/trials*100)
        print("%0.2f%% trials had <= 1 symmetric difference" % (num_symm_diff/trials*100))
        arr_symm_diff.append(num_symm_diff/trials*100)
        print("%0.2f%% trials had <= 2 symmetric difference" % (num_symm_diff_2/trials*100))
        arr_symm_diff_2.append(num_symm_diff_2/trials*100)
    
    plt.plot(sample_range, arr_perfectly_chosen)
    plt.plot(sample_range, arr_predictors_missed)
    plt.plot(sample_range, arr_false_predictors_chosen)
    plt.plot(sample_range, arr_symm_diff)
    plt.plot(sample_range, arr_symm_diff_2)
    plt.legend(["% that perfectly chose predictors", "% that missed <= 1 predictors", "% that chose <= 1 false predictors", "% that had <= 1 symmetric difference", "% that had <= 2 symmetric difference"])
    plt.ylim([0,101])
    plt.ylabel('%')
    plt.xlabel('Number of data samples')
    plt.show()


def plot_mse_prediction(df, sample_range, trials):
    """
    Plots the prediction accuracy of lasso for different training set sizes
    """
    mse = []
    
    for num_samples in sample_range:
        sum_sq_err = 0.0
        print(num_samples)
        for t in range(trials):
            train_data = get_distribution_samples(df, num_samples)
            test_data = get_distribution_samples(df, 1)
            model = Lasso(alpha=find_best_alpha_mse(train_data))
            train_x = train_data.drop(["y"], axis=1)
            test_x = test_data.drop(["y"], axis=1)
            model.fit(train_x, train_data[["y"]])
            sum_sq_err += (test_data.loc[0, 'y'] - model.predict(test_x)[0])**2
        mse.append(sum_sq_err/trials)
        
    plt.figure()
    plt.plot(sample_range, mse)
    plt.ylabel('MSE')
    plt.xlabel('Number of data samples')
    plt.show()
    
if __name__ == "__main__":
    trials = 50 
    sample_range = range(10, 40, 5)
    df = pd.read_csv("data.csv")
    #plot_mse_prediction(df, sample_range, trials)
    plot_subset_accuracy(df, sample_range, trials)
#plt.matshow(df.corr())
#plt.suptitle('Original data')
#plt.show()

#plt.matshow(sampled_data.corr())
#plt.suptitle('Sampled data')
#plt.show()

#scatter_matrix(df, diagonal='kde')
#plt.suptitle('Original data')
#plt.show()
#scatter_matrix(sampled_data, diagonal='kde')
#plt.suptitle('Sampled data')
#plt.show()