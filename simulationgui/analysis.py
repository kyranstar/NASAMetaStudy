from multiprocessing import Pool
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsIC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from datasampling import get_distribution_samples
from math import *
import utility


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
    return set([i+1 for i, x in enumerate(coef) if x != 0.0])


def lasso_mse(data, dependent_var, test_data, alphas=np.arange(0.05, 1.0, 0.05), stddev=0):
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
        # model.fit(predictors, data[['y']])
            # create and train model in cv
        model = Lasso(alpha=alpha)
        model.fit(trainx, trainy)
        score = mean_squared_error(testy, model.predict(testx))
        return score
    trainx = data.drop([dependent_var], axis=1)
    trainy = data[[dependent_var]]
    testx = test_data.drop([dependent_var], axis=1)
    testy = test_data[[dependent_var]]
    # print(list(map(lambda a: mse_prediction(a), alphas)))
    if stddev == 0:
        return min(alphas, key=lambda alpha: mse_prediction(alpha, trainx, trainy, testx, testy))
    scores = np.array([mse_prediction(a, trainx, trainy, testx, testy) for a in alphas])
    std = scores.std()*stddev
    minscore = scores.min()
    # Remove all alphas whose mse is above one standard deviation from min mse
    best_alpha = np.array([a[0] for a in zip(alphas, scores) if a[1] <= minscore + std]).max()

    model = Lasso(alpha=best_alpha)
    model.fit(data.drop([dependent_var], axis=1), data[[dependent_var]])

    return model


def lasso_cv_mse(data, dependent_var, cv=10, alphas=np.arange(0.01, 1.0, 0.05), stddev=0):
    """
    Attempts to find the best lasso alpha value using generalized prediction mse.
    It first finds the CV MSE prediction scores associated with each alpha value,
    and returns the largest alpha value within one standard deviation of the minimum
    prediction score. This reduces generalization error from variance that would occur
    from just selecting the minimum prediction score.
    """
    def mse_prediction(alpha, data):
        # model.fit(predictors, data[['y']])
        score = 0.0
        for _ in range(cv):
            # create and train model in cv
            model = Lasso(alpha=alpha)
            train, test = train_test_split(data, test_size=1.0/cv)

            trainx = train.drop([dependent_var], axis=1)
            trainy = train[[dependent_var]]
            model.fit(trainx, trainy)

            testx = test.drop([dependent_var], axis=1)
            testy = test[[dependent_var]]
            score += mean_squared_error(testy, model.predict(testx))

        return score/cv

    if stddev == 0:
        best_alpha = min(alphas, key=lambda alpha: mse_prediction(alpha, data))
    else:
        scores = np.array([mse_prediction(a, data) for a in alphas])
        std = scores.std()*stddev
        minscore = scores.min()
        # Remove all alphas whose mse is above one standard deviation from min mse
        best_alpha = np.array([a[0] for a in zip(alphas, scores) if a[1] <= minscore + std]).max()

    model = Lasso(alpha=best_alpha)
    model.fit(data.drop([dependent_var], axis=1), data[[dependent_var]])

    return model


def lasso_bic(data, dependent_var):
    """
    Attempts to find the best lasso alpha value fitting a dataset using the BIC
    metric: https://stats.stackexchange.com/questions/126898/tuning-alpha-parameter-in-lasso-linear-model-in-scikitlearn
    """
    predictors = data.drop([dependent_var], axis=1)
    # def bic_score(predictors, y, model):
    #    sse = sum((model.predict(predictors) - y.values[0])**2)
    #    s = np.count_nonzero(model.coef_)
    #    n = len(predictors.columns)
    #    cn = math.sqrt(n)/(s*s)
    #    print(math.log(sse/n) + s*math.log(n)/n*cn)
    #    return math.log(sse/n) + abs(s)*math.log(n)/n*cn

    model = LassoLarsIC(criterion='bic')
    model.fit(predictors, data[[dependent_var]])
    return model


def random_forest(data, dependent_var):
    # TODO
    predictors = data.drop([dependent_var], axis=1)
    # def bic_score(predictors, y, model):
    #    sse = sum((model.predict(predictors) - y.values[0])**2)
    #    s = np.count_nonzero(model.coef_)
    #    n = len(predictors.columns)
    #    cn = math.sqrt(n)/(s*s)
    #    print(math.log(sse/n) + s*math.log(n)/n*cn)
    #    return math.log(sse/n) + abs(s)*math.log(n)/n*cn

    model = RandomForestRegressor()
    model.fit(predictors, data[[dependent_var]].values.ravel())
    return model


def predict_worker_func(method, true_model_text, data_model, trials, num_samples):
    model_generator = PREDICT_FUNCS[method]
    sum_sq_err = 0.0

    for trialnum in range(trials):
        print("Trial %d" % trialnum)
        sampled_data = get_distribution_samples(data_model, num_samples, true_model_text)

        model = model_generator(sampled_data, data_model.dependent_var)

        test_data = get_distribution_samples(data_model, 1, true_model_text)
        # print(test_data)
        sum_sq_err += (test_data.loc[0, data_model.dependent_var] -
                       model.predict(test_data.drop([data_model.dependent_var], axis=1))[0])**2

    return (sum_sq_err/trials)


def worker_func(method, true_model_text, data_model, trials, num_samples, true_variables):

    model_generator = SUBSET_FUNCS[method]
    sum_sq_err = 0.0
    num_perfectly_chosen = 0.0
    num_predictors_missed = 0.0
    num_false_predictors_chosen = 0.0
    num_symm_diff = 0.0
    num_symm_diff_2 = 0.0
    ave_symm_diff = 0.0

    for trialnum in range(trials):
        print("Trial %d, num_samples: %d" % (trialnum, num_samples))
        sampled_data = get_distribution_samples(data_model, num_samples, true_model_text)

        model = model_generator(sampled_data, data_model.dependent_var)

        chosen_variables = select_variables(model)
        if chosen_variables != true_variables:
            try:
                print([data_model.variables[i-1] for i in chosen_variables])
            except:
                pass

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
        test_data = get_distribution_samples(data_model, 1, true_model_text)
        # print(test_data)
        sum_sq_err += (test_data.loc[0, data_model.dependent_var] -
                       model.predict(test_data.drop([data_model.dependent_var], axis=1))[0])**2

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


def subset_accuracy(variables, data_model, true_model_text, sample_range, trials, subset_metrics, subset_methods, predict_methods, error_types):
    """
    Plots various metrics of how accurately lasso can select the true variables
    given certain numbers of samples.
    Arguments:
        df: The dataset
        sample_range: A list of numbers of samples to try (x-axis values)
        trials: The number of trials to do for each sample size
    """
    formula_variables = utility.extract_variables(true_model_text, variables)
    true_variables = set([i+1 for i, v in enumerate(variables)
                          if v in formula_variables])

    output_data = pd.DataFrame({'sample_size': sample_range})

    pool = Pool(processes=8)

    for method in predict_methods:
        mse = []
        processes = [pool.apply_async(predict_worker_func, (method, true_model_text, data_model,
                                                            trials, num_samples)) for num_samples in sample_range]
        for mean_sq_err in [p.get() for p in processes]:
            mse.append(mean_sq_err)
        output_data[(method, 'prediction_mse')] = mse

    for method in subset_methods:

        arr_perfectly_chosen = []
        arr_predictors_missed = []
        arr_false_predictors_chosen = []
        arr_symm_diff = []
        arr_symm_diff_2 = []
        arr_ave_symm_diff = []
        mse = []

        processes = [pool.apply_async(worker_func, (method, true_model_text, data_model,
                                                    trials, num_samples, true_variables)) for num_samples in sample_range]
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


SUBSET_FUNCS = {'LassoCV': lasso_cv_mse,
                'LassoCVStd': lambda data, dependent_var: lasso_cv_mse(data, dependent_var, stddev=1.0),
                'LassoBIC': lasso_bic, }
PREDICT_FUNCS = {'RandomForest': random_forest}
