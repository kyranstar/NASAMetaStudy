import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsIC
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import datasampling
from multiprocessing import Pool

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
    #def bic_score(predictors, y, model):
    #    sse = sum((model.predict(predictors) - y.values[0])**2)
    #    s = np.count_nonzero(model.coef_)   
    #    n = len(predictors.columns)
    #    cn = math.sqrt(n)/(s*s)
    #    print(math.log(sse/n) + s*math.log(n)/n*cn)
    #    return math.log(sse/n) + abs(s)*math.log(n)/n*cn

    model = LassoLarsIC(criterion='bic')
    model.fit(predictors, data[['y']])
    return model

def worker_func(method, true_model, df, trials, num_samples, true_variables):
    
    model_generator = SUBSET_FUNCS[method]
    sum_sq_err = 0.0
    num_perfectly_chosen = 0.0
    num_predictors_missed = 0.0
    num_false_predictors_chosen = 0.0
    num_symm_diff = 0.0
    num_symm_diff_2 = 0.0
    ave_symm_diff = 0.0
    
    for t in range(trials):    
        sampled_data = datasampling.get_distribution_samples(df, num_samples, true_model)

        model = model_generator(sampled_data)

        chosen_variables = select_variables(model)

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
    print("%0.2f%% trials missed <= 1 predictors" % (num_predictors_missed/trials*100))
    print("%0.2f%% trials had <= 1 false predictors" % (num_false_predictors_chosen/trials*100))
    print("%0.2f%% trials had <= 1 symmetric difference" % (num_symm_diff/trials*100))
    print("%0.2f%% trials had <= 2 symmetric difference" % (num_symm_diff_2/trials*100))
    print("%0.2f is average symmetric difference" % (ave_symm_diff/trials))
    
    return (num_samples,
            num_perfectly_chosen/trials*100,
            num_predictors_missed/trials*100,
            num_false_predictors_chosen/trials*100,
            num_symm_diff/trials*100,
            num_symm_diff_2/trials*100,
            ave_symm_diff/trials,
            sum_sq_err/trials)

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
    
    pool = Pool(processes=4)
    
    for method in subset_methods:
            
        arr_perfectly_chosen = []
        arr_predictors_missed = []
        arr_false_predictors_chosen = []
        arr_symm_diff = []
        arr_symm_diff_2 = []
        arr_ave_symm_diff = []
        mse = []
        
        processes = [pool.apply_async(worker_func, (method, true_model, df, trials, num_samples, true_variables)) for num_samples in sample_range]
        
        for (num_samples,
             num_perfectly_chosen,
             num_predictors_missed,
             num_false_predictors_chosen,
             num_symm_diff,
             num_symm_diff_2,
             ave_symm_diff,
             sum_sq_err) in [p.get() for p in processes]:
            
            arr_perfectly_chosen.append(num_perfectly_chosen)
            arr_predictors_missed.append(num_predictors_missed)
            arr_false_predictors_chosen.append(num_false_predictors_chosen)
            arr_symm_diff.append(num_symm_diff)
            arr_symm_diff_2.append(num_symm_diff_2)
            arr_ave_symm_diff.append(ave_symm_diff)
            mse.append(sum_sq_err)
            
        output_data[(method, 'perfectly_chosen')] = arr_perfectly_chosen
        output_data[(method, 'predictors_missed')] = arr_predictors_missed
        output_data[(method, 'false_predictors_chosen')] = arr_predictors_missed
        output_data[(method, 'symm_diff')] = arr_symm_diff
        output_data[(method, 'symm_diff_2')] = arr_symm_diff_2
        output_data[(method, 'ave_symm_diff')] = arr_ave_symm_diff
        output_data[(method, 'prediction_mse')] = mse
    pool.close()
    return output_data