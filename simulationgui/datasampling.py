import math
import numpy as np
import pandas as pd


def get_distribution_samples(data, num_samples, true_model, y_variance=0.0):
    """
    Given a set of normally distributed data, creates num_samples points of
    simulated data.
    """
    # Generate xs as multivariate normal variables
    predictors = data.drop(['y'], axis=1)
    mean = predictors.mean(axis=0)
    cov = predictors.corr()
    samples = np.random.multivariate_normal(mean, cov, size=num_samples)

    samples_df = pd.DataFrame(data=samples, columns=predictors.columns)

    # Calculate the error term in the regression
    # if not y_variance:
    #    sse = np.square(data[['y']].values - true_model.predict(xs)).sum()
    #    y_variance = sse/(data.shape[0] - data.shape[1])#np.array(data[['y']].values - true_model.predict(xs)).var()
    #    print(y_variance)
    samples_df.loc[:, 'y'] = true_model.predict(samples) + \
        np.random.normal(0, math.sqrt(y_variance), num_samples)
    return samples_df
