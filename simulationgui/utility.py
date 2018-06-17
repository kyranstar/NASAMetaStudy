import dash_html_components as html
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
import scipy.stats as stats
import pandas as pd
from gen_ui import Ui_MainWindow


def get_super_parent(ob):
    curr_ob = ob.parentWidget()
    while not isinstance(curr_ob, Ui_MainWindow) and not curr_ob is None:
        curr_ob = curr_ob.parentWidget()
    return curr_ob


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


def dataframe_to_tr(data, editable_cols):
    """
    Converts a pandas dataframe to a html table rows.
    Arguments:
        data: The dataframe
        editable_cols: The columns to make editable
    Returns:
        A list of html.Tr objects
    """
    header_row = [html.Tr([html.Th(col) for col in data.columns])]
    body = [html.Tr([
        html.Td(data.iloc[i][col], contentEditable=(col in editable_cols)) for col in data.columns
    ]) for i in range(len(data))]

    return header_row + body


def tr_to_dataframe(rows):
    rows = list(map(lambda row: [element['props']['children']
                                 for element in row['props']['children']], rows))
    print(rows)
    cols = [th for th in rows[0]]
    body = [[
        td for td in row
    ] for row in rows[1:]]

    return pd.DataFrame(body, columns=cols)
