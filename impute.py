
import numpy as np
import pandas as pd
import impyute as impy
import itertools
import matplotlib.pyplot as plt
from sklearn import preprocessing


def scatterplot_matrix(old_data, imputed_data, names, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numdata, numvars = imputed_data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i, j), (j, i)]:
            for z in range(numdata):
                col = 'red' if np.isnan(old_data.ix[z, x]) or np.isnan(
                    old_data.ix[z, y]) else 'black'
                axes[x, y].scatter(imputed_data.ix[z, x], imputed_data.ix[z, y],
                                   color=col, **kwargs)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                            ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        axes[i, j].yaxis.set_visible(True)

    return fig


if __name__ == "__main__":
    df = pd.read_csv('real_data.csv')
    pruned_df = df.drop(['x15', 'x16', 'x17', 'x18'], axis=1)
    data = pruned_df.values

    # Normalizer
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    mice_impute = pd.DataFrame(min_max_scaler.fit_transform(impy.mice(data.copy())),
                               index=pruned_df.index, columns=pruned_df.columns)

    scatterplot_matrix(pruned_df, mice_impute, pruned_df.columns)
    mice_impute['x15'] = df['x15']
    mice_impute['x16'] = df['x16']
    mice_impute['x17'] = df['x17']
    mice_impute['x18'] = df['x18']
    print(mice_impute)
    mice_impute.to_csv('mice_real_data.csv')
    # plt.show()
