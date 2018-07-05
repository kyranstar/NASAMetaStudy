import importlib
import random
import numpy as np
import pandas as pd
from utility import calculate_dummy
from scipy.stats import norm
import math

_math_funcs = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh', 'degrees',
               'e', 'exp', 'fabs', 'floor', 'fmod', 'frexp', 'hypot', 'ldexp', 'log',
               'log10', 'modf', 'pi', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']
_rand_funcs = ['random']

# use the list to filter the local namespace
math_module = importlib.import_module('math')
allowed_funcs = dict([(k, getattr(math_module, k)) for k in _math_funcs])
rand_module = importlib.import_module('random')
allowed_funcs = dict(allowed_funcs, **dict([(k, getattr(rand_module, k)) for k in _rand_funcs]))
# add any needed builtins back in.
allowed_funcs['abs'] = abs


class DataModel():
    def __init__(self, mean, cov, variables, cat_portions, dummy_cols, dependent_var):
        """
        cov: The covariance matrix
        cat_portions: a map of the categorical columns to a list of tuples, where the
            first entry is a categorical value and the second is the fraction of the time
            they show up in the data.
        """
        self.num_samples = len(mean)
        self.mean = mean
        self.cov = cov
        self.variables = variables
        self.cat_portions = cat_portions
        self.dummy_cols = dummy_cols
        self.dependent_var = dependent_var


def true_model(predictors, dependent_var, true_model_text):
    result = pd.DataFrame(0, index=np.arange(len(predictors)), columns=[dependent_var])

    for i in range(len(predictors)):
        scope = allowed_funcs
        for col in predictors.columns:
            scope[col] = predictors.loc[i, col]
        result.loc[i, dependent_var] = eval(true_model_text, {"__builtins__": None}, scope)
    return result


def get_distribution_samples(data_model, num_samples, true_model_text):
    """
    Given a set of normally distributed data, creates num_samples points of
    simulated data.
    """

    # Generate xs as multivariate normal variables
    samples = np.random.multivariate_normal(
        data_model.mean, data_model.cov, size=num_samples)

    samples_df = pd.DataFrame(data=samples, columns=data_model.variables)

    for cat_col, portions in data_model.cat_portions.items():
        # Calculate inverse cumulative thresholds
        thresholds = []
        cumulative_portion = 0.0
        mean = data_model.mean[data_model.variables.index(cat_col)]
        stddev = math.sqrt(data_model.cov.loc[cat_col, cat_col])
        for val, frac in portions:
            cumulative_portion += frac
            threshold = norm.ppf(
                cumulative_portion, loc=mean, scale=stddev)
            thresholds.append((val, threshold))
        for i, norm_value in enumerate(samples_df[cat_col]):
            for val, threshold in thresholds:
                if norm_value < threshold:
                    samples_df.loc[i, cat_col] = val
                    break
    #samples_df = calculate_dummy(samples_df, data_model.cat_portions.keys(), data_model.dummy_cols)
    # Calculate the error term in the regression
    # if not y_variance:
    #    sse = np.square(data[['y']].values - true_model.predict(xs)).sum()
    #    y_variance = sse/(data.shape[0] - data.shape[1])#np.array(data[['y']].values - true_model.predict(xs)).var()
    #    print(y_variance)
    samples_df.loc[:, data_model.dependent_var] = true_model(
        samples_df, data_model.dependent_var, true_model_text)
    return samples_df
