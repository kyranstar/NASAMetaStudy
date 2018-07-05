import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.covariance import OAS
import scipy.stats as stats
import pandas as pd
from PyQt5 import QtWidgets
from scipy.stats import norm, truncnorm


def correlations(df, categorical_portions):
    categorical_cols = list(categorical_portions.keys())
    # First generate continuous samples for categorical values. We do this by sampling from
    # a truncated normal distribution in the range for that continous variable.
    for categorical_col in categorical_cols:
        portions = categorical_portions[categorical_col]
        # The values of the categorical variable in order
        portions_keys = [val for val, frac in portions]
        for i, cat_val in enumerate(df[categorical_col]):
            if len(portions) == 1:
                df.loc[i, categorical_col] = norm.rvs()
                # Normal sample
                continue
            ind = portions_keys.index(cat_val)
            # Get sums of prev portions including and not including this portion
            sum_a = sum(map(lambda i: portions[i][1], range(ind)))
            sum_b = sum_a + portions[ind][1]
            # Get thresholds
            threshold_a = norm.ppf(sum_a, loc=0.0, scale=1.0)
            threshold_b = norm.ppf(sum_b, loc=0.0, scale=1.0)
            # Sample truncated norm
            df.loc[i, categorical_col] = truncnorm.rvs(threshold_a, threshold_b)

    # estimate covariance matrix
    estimator = OAS()
    estimator.fit(df.values)

    cov = pd.DataFrame(estimator.covariance_,
                       index=df.columns, columns=df.columns)

    #cov = cov[orig_columns].reindex(orig_columns)

    return cov


def error(msg):
    error_dialog = QtWidgets.QErrorMessage()
    error_dialog.showMessage(msg)
    error_dialog.exec_()


def calculate_dummy(converting_df, categorical_cols, dummy_cols=None):
    """
    Converts categorical columns to dummy columns.
    Arguments:
        converting_df: The dataframe to convert
        categorical_cols: The categorical columns in converting_df to convert
        dummy_cols: The dummy columns to always have in the returned dataframe,
            even if the values didn't show up in converting_df.
    """
    categorical_cols = set(categorical_cols).intersection(set(converting_df.columns))
    # Dummy encode the new df
    for categorical_col in categorical_cols:
        dummies = pd.get_dummies(converting_df[categorical_col]).rename(
            columns=lambda x: '%s_%s' % (str(categorical_col), str(x)))
        converting_df = pd.concat([converting_df, dummies], axis=1)
        converting_df = converting_df.drop([categorical_col], axis=1)
    if not dummy_cols is None:
        # Add missing columns where values didn't show
        missing_cols = set(dummy_cols) - set(converting_df.columns)
        for c in missing_cols:
            converting_df[c] = 0

        assert(set(dummy_cols) - set(converting_df.columns) == set())

    return converting_df


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
    plt.plot(h, fit, '-o')
    plt.hist(h, normed=True)  # use this to draw histogram of your data
    plt.show()  # use may also need add this
