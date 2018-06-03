import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsIC
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import train_test_split
import datasampling

SUBSET_FUNCS = {'LassoCV': lambda data: lasso_cv_mse(data), 
                'LassoCVStd': lambda data: lasso_cv_mse(data, stddev=1.0),
                'LassoBIC': lambda data: lasso_bic(data),}

def select_variables(model):
    """
    Given a model, fits it to the data and gives a set of the 1-based indices of 
    non-zero regression coefficients. 
    Arguments:
        model: Some regression model, like Lasso or LinearRegression
        data: Data to fit the model to, where the input features are all columns
            but "y" and the output column is "y"
    Example:
        If regression coefficients are [0.5, 0.4, 0.0, 3.0], returns {1, 2, 4}
    """
    coef = list(np.array(model.coef_).flat)
    return set([i+1 for i,x in enumerate(coef) if x != 0.0])
    
def lasso_mse(data, test_data, alphas=np.arange(0.05,1.0,0.05), stddev=0):
    """
    Attempts to find the best lasso alpha value using generalized prediction mse.
    It first finds the CV MSE prediction scores associated with each alpha value,
    and returns the largest alpha value within stddev standard deviations of the minimum
    prediction score. If 0, this just returns the minimum prediction score. For 
    larger values, this might reduce generalization error from variance that would occur
    from just selecting the minimum prediction score.
    
    Arguments:
        data:
        test_data:
        alphas:
        stddev:
    """
    def mse_prediction(alpha, trainx, trainy, testx, testy):
        #model.fit(predictors, data[['y']])
            # create and train model in cv
        model = Lasso(alpha=alpha)
        model.fit(trainx, trainy)
        score = mean_squared_error(testy, model.predict(testx))
        return score
    trainx = data.drop(["y"], axis=1)
    trainy = data[['y']]
    testx = test_data.drop(['y'], axis=1)
    testy = test_data[['y']]
    #print(list(map(lambda a: mse_prediction(a), alphas)))
    if stddev == 0:
        return min(alphas, key=lambda alpha: mse_prediction(alpha, trainx, trainy, testx, testy))
    scores = np.array([mse_prediction(a, trainx, trainy, testx, testy) for a in alphas])
    std = scores.std()*stddev
    minscore = scores.min()
    # Remove all alphas whose mse is above one standard deviation from min mse
    best_alpha = np.array([a[0] for a in zip(alphas, scores) if a[1] <= minscore + std]).max()

    model = Lasso(alpha=best_alpha)
    model.fit(data.drop(["y"], axis=1), data[['y']])

    return model

def lasso_cv_mse(data, cv=10, alphas=np.arange(0.01,1.0,0.05), stddev=0):
    """
    Attempts to find the best lasso alpha value using generalized prediction mse.
    It first finds the CV MSE prediction scores associated with each alpha value,
    and returns the largest alpha value within one standard deviation of the minimum
    prediction score. This reduces generalization error from variance that would occur
    from just selecting the minimum prediction score.
    """
    def mse_prediction(alpha, data):
        #model.fit(predictors, data[['y']])
        score = 0.0
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
        
        return score/cv
    
    if stddev == 0:
        best_alpha = min(alphas, key=lambda alpha: mse_prediction(alpha,data))
    else:
        scores = np.array([mse_prediction(a, data) for a in alphas])
        std = scores.std()*stddev
        minscore = scores.min()
        # Remove all alphas whose mse is above one standard deviation from min mse
        best_alpha = np.array([a[0] for a in zip(alphas, scores) if a[1] <= minscore + std]).max()
    
    model = Lasso(alpha=best_alpha)
    model.fit(data.drop(["y"], axis=1), data[['y']])

    return model

def lasso_bic(data):
    """
    Attempts to find the best lasso alpha value fitting a dataset using the BIC
    metric: https://stats.stackexchange.com/questions/126898/tuning-alpha-parameter-in-lasso-linear-model-in-scikitlearn
    """
    predictors = data.drop(["y"], axis=1)
    #alphas = np.arange(0.1, 1.0, 0.05)
    #best_score = np.inf
    #best_alpha = 0
    def bic_score(predictors, y, model):
        sse = sum((model.predict(predictors) - y.values[0])**2)
        s = np.count_nonzero(model.coef_)   
        n = len(predictors.columns)
        cn = math.sqrt(n)/(s*s)
        print(math.log(sse/n) + s*math.log(n)/n*cn)
        return math.log(sse/n) + abs(s)*math.log(n)/n*cn
        
    #for alpha in alphas:
    #    model = Lasso(alpha=alpha)
    #    model.fit(predictors, data[['y']])
    #    score = bic_score(predictors, data[['y']], model)
    #    if score < best_score:
    #        best_score = score
    #        best_alpha = alpha
    model = LassoLarsIC(criterion='bic')
    model.fit(predictors, data[['y']])
    return model

def subset_accuracy(df, sample_range, trials, subset_metrics, subset_methods, error_types):
    """
    Plots various metrics of how accurately lasso can select the true variables
    given certain numbers of samples.
    Arguments:
        df: The dataset
        sample_range: A list of numbers of samples to try (x-axis values)
        trials: The number of trials to do for each sample size
    """
    true_model = Lasso(alpha=.5)    
    true_model.fit(df.drop(["y"], axis=1), df[['y']])
    true_variables = select_variables(true_model)
    print(true_model.coef_)
    print(true_model.intercept_)
    
    output_data = pd.DataFrame({'sample_size':sample_range})
    
    for method in subset_methods:
        arr_perfectly_chosen = []
        arr_predictors_missed = []
        arr_false_predictors_chosen = []
        arr_symm_diff = []
        arr_symm_diff_2 = []
        arr_ave_symm_diff = []
        mse = []
        
        for num_samples in sample_range:
            
            sum_sq_err = 0.0
            num_perfectly_chosen = 0.0
            num_predictors_missed = 0.0
            num_false_predictors_chosen = 0.0
            num_symm_diff = 0.0
            num_symm_diff_2 = 0.0
            ave_symm_diff = 0.0
            for t in range(trials):    
                sampled_data = datasampling.get_distribution_samples(df, num_samples, true_model)

                model = SUBSET_FUNCS[method](sampled_data)

                chosen_variables = select_variables(model)
                #print(chosen_variables)
                #print(chosen_variables)
                #print(chosen_variables)
                #pause
                #print(best_alpha)
                #print(len(chosen_variables))
                symm_diff = len(true_variables.symmetric_difference(chosen_variables))
                if chosen_variables == true_variables:
                    num_perfectly_chosen += 1
                if len(chosen_variables) - len(true_variables.intersection(chosen_variables)) <= 1:
                    num_predictors_missed += 1
                if len(true_variables) - len(true_variables.intersection(chosen_variables)) <= 1:
                    num_false_predictors_chosen += 1
                if symm_diff <= 1:
                    num_symm_diff += 1
                if symm_diff <= 2:
                    num_symm_diff_2 += 1
                ave_symm_diff += symm_diff
                test_data = datasampling.get_distribution_samples(df, 1, true_model)
                sum_sq_err += (test_data.loc[0, 'y'] - model.predict(test_data.drop(["y"], axis=1))[0])**2
            
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
            print("%0.2f is average symmetric difference" % (ave_symm_diff/trials))
            arr_ave_symm_diff.append(ave_symm_diff/trials)
            mse.append(sum_sq_err/trials)
        output_data[(method, 'perfectly_chosen')] = arr_perfectly_chosen
        output_data[(method, 'predictors_missed')] = arr_predictors_missed
        output_data[(method, 'false_predictors_chosen')] = arr_predictors_missed
        output_data[(method, 'symm_diff')] = arr_symm_diff
        output_data[(method, 'symm_diff_2')] = arr_symm_diff_2
        output_data[(method, 'ave_symm_diff')] = arr_ave_symm_diff
        output_data[(method, 'prediction_mse')] = mse
    return output_data
    
if __name__ == "__main__":
    trials = 400
    sample_range = range(10,300,50)
    df = pd.read_csv("data.csv")
    #plot_lasso(df)
    #plot_mse_prediction(df, sample_range, trials)
    #plot_subset_accuracy(df, sample_range, trials)
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