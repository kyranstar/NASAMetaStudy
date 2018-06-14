import math
import numpy as np
import pandas as pd


class DataModel():
    def __init__(self, mean, cov, variables):
        self.num_samples = len(mean)
        self.mean = mean
        self.cov = cov
        self.variables = variables


def true_model(predictors, true_model_text):
    result = pd.DataFrame(0, index=np.arange(len(predictors)), columns=['y'])

    safe_list = ['math', 'acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh', 'degrees',
                 'e', 'exp', 'fabs', 'floor', 'fmod', 'frexp', 'hypot', 'ldexp', 'log',
                 'log10', 'modf', 'pi', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']

    # use the list to filter the local namespace
    allowed_funcs = dict([(k, locals().get(k, None)) for k in safe_list])
    # add any needed builtins back in.
    allowed_funcs['abs'] = abs

    for i in range(len(predictors)):
        scope = allowed_funcs
        for col in predictors.columns:
            scope[col] = predictors.loc[i, col]

        result.loc[i, 'y'] = eval(true_model_text, {"__builtins__": None}, scope)
    return result


def get_distribution_samples(data_model, num_samples, true_model_text):
    """
    Given a set of normally distributed data, creates num_samples points of
    simulated data.
    """
    # Generate xs as multivariate normal variables
    samples = np.random.multivariate_normal(
        data_model.mean, data_model.cov, size=data_model.num_samples)

    samples_df = pd.DataFrame(data=samples, columns=data_model.variables)

    # Calculate the error term in the regression
    # if not y_variance:
    #    sse = np.square(data[['y']].values - true_model.predict(xs)).sum()
    #    y_variance = sse/(data.shape[0] - data.shape[1])#np.array(data[['y']].values - true_model.predict(xs)).var()
    #    print(y_variance)
    samples_df.loc[:, 'y'] = true_model(samples_df, true_model_text)
    return samples_df
