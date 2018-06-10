from ast import literal_eval
import json
from dash.dependencies import Input, Output, State, Event
import pandas as pd
import plotly.graph_objs as go

from .app import ERROR_TYPES, SUBSET_METRICS
from .analysis import subset_accuracy
from .server import app


@app.callback(Output('graph-data-df', 'children'),
              state=[State('distributions-table', 'rows'),
                     State('num-trials', 'value'),
                     State('samples-min', 'value'),
                     State('samples-max', 'value'),
                     State('samples-step', 'value'),
                     State('variable-selection-metrics', 'value'),
                     State('variable-selection-methods', 'value'),
                     State('error-types', 'value')],
              events=[Event('run-button', 'click')])
def update_graph_data(data_dict, trials, samples_min, samples_max, samples_step, subset_metrics, subset_methods, error_types):
    """
    Runs the experiment; updates the internal json representation of the results when
    the run button is clicked.
    """
    if None in [trials, samples_min, samples_max, samples_step, subset_metrics, subset_methods, error_types]:
        return pd.DataFrame().to_json()
    data = pd.DataFrame(data_dict)
    sample_range = range(int(samples_min), int(samples_max), int(samples_step))
    data_df = subset_accuracy(data, sample_range, int(
        trials), subset_metrics, subset_methods, error_types)
    # convert dataframe to nested dictionary
    df_dict = data_df.groupby(level=0).apply(lambda df: df.xs(df.name).to_dict()).to_dict()
    # convert tuple keys to strings
    df_dict = {str(i): {str(key): inner_val for key, inner_val in val.items()}
               for i, val in df_dict.items()}
    # convert to json
    json_str = json.dumps(df_dict)
    return json_str


def load_graph_data(cached_df):
    """
    Loads the result of the experiment from json to a dataframe. This needs a
    special function because we are using tuple indices.
    """
    # load json
    df_dict = json.loads(cached_df)

    def eval_tuple(strobj):
        if strobj.startswith('('):
            return literal_eval(strobj)
        return strobj
    # convert tuple strings to python tuples
    df_dict = {int(i): {eval_tuple(t): val for t, val in v.items()} for i, v in df_dict.items()}
    # flip rows and columns and convert to dataframe
    return pd.DataFrame.from_dict(df_dict).T


@app.callback(Output('prediction-error-graph', 'figure'),
              [Input('graph-data-df', 'children')])
def prediction_error_update_graph(cached_graph_df):
    """
    Update the prediction error graph from the results of the experiment.
    """
    if cached_graph_df is None:
        return {'data': []}
    graph_data = load_graph_data(cached_graph_df)
    error_type_values = [s['value'] for s in ERROR_TYPES]
    plot_data = []
    for (subset_method, subset_metric) in [col for col in graph_data.columns if col[1] in error_type_values]:
        fig = go.Scatter(
            x=graph_data['sample_size'].values,
            y=graph_data[(subset_method, subset_metric)].values,
            mode='lines+markers',
            marker={'size': 8},
            name="%s: %s" % (subset_method, subset_metric))
        plot_data.append(fig)
    return {'data': plot_data,
            'layout': go.Layout(
                xaxis={'title': 'Number of samples'},
                yaxis={'title': '%'},
                # hovermode='closest'
            )}


@app.callback(Output('variable-selection-graph', 'figure'),
              [Input('graph-data-df', 'children')])
def variable_selection_update_graph(cached_df):
    """
    Updates the variable selection graph from the results of the experiment.
    """
    if cached_df is None:
        return {'data': []}
    graph_data = load_graph_data(cached_df)
    plot_data = []
    subset_metric_values = [s['value'] for s in SUBSET_METRICS]
    # Get the columns that we are going to graph
    subset_cols = [col for col in graph_data.columns if col[1] in subset_metric_values]
    for (subset_method, subset_metric) in subset_cols:
        fig = go.Scatter(
            x=graph_data['sample_size'].values,
            y=graph_data[(subset_method, subset_metric)].values,
            mode='lines+markers',
            marker={'size': 8},
            name="%s: %s" % (subset_method, subset_metric))
        plot_data.append(fig)
    return {'data': plot_data,
            'layout': go.Layout(
                xaxis={'title': 'Number of samples'},
                yaxis={'title': '%'},
                # hovermode='closest'
            )}
