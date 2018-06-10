import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import pandas as pd
from .server import app

#from .server import app, server

DISTRIBUTION_COLUMNS = ["Name", "Type", "Mean", "Variance"]
SUBSET_METRICS = [
    {'label': 'perfectly_chosen', 'value': 'perfectly_chosen'},
    {'label': 'predictors_missed', 'value': 'predictors_missed'},
    {'label': 'false_predictors_chosen', 'value': 'false_predictors_chosen'},
    {'label': 'symm_diff', 'value': 'symm_diff'},
    {'label': 'symm_diff_2', 'value': 'symm_diff_2'},
]
ERROR_TYPES = [
    {'label': 'Prediction MSE', 'value': 'prediction_mse'},
    {'label': 'Prediction Mean Error', 'value': 'prediction_me'},
]


def data_tab():
    """
    Construct the initial data tab in the layout.
    """
    return html.Div([
        # Data upload
        html.Div([
            dcc.Upload(html.Button('Upload Data', id='upload-data-button', style={'width': '100%'}),
                       id='upload-data',
                       className='eight columns'),
        ], className='row'),
        html.Div(id='input-data-df', style={'display': 'none'}),
        # Labels row
        html.Div([
            html.Label('Distributions', className='eight columns'),
            html.Label('True Model', className='four columns')
        ], className='row'),
        html.Div([
            html.Div([
                dt.DataTable(
                    rows=pd.DataFrame(columns=DISTRIBUTION_COLUMNS).to_dict('records'),
                    columns=DISTRIBUTION_COLUMNS,
                    editable=True,
                    id='distributions-table'),
                html.Div([], id='distributions-table-df', style={'display': 'none'}),
                html.Button("Add Variable",
                            id='add-variable-button',
                            n_clicks=0,
                            style={
                                'width': '100%'
                            }),
                html.Button("Fit Distributions",
                            id='fit-distributions-button',
                            n_clicks=0,
                            style={
                                'width': '100%'
                            })
            ], className='eight columns'),
            html.Div([],
                     className='four columns',
                     id='true-model-table')
        ], className='row'),
        html.Div([
            html.Label('Correlations'),
        ], className='row'),
        html.Div([
            html.Div([
                dt.DataTable(rows=[{}],
                             columns=['Name'],
                             id='correlation-table'),
            ], className='eight columns'),
        ], className='row'),
        html.Div([
            html.Div([
                dcc.Graph(id='correlation-heatmap'),
            ], className='eight columns')
        ], className='row'),
    ], className="page", style={'display': 'none'}, id='data-tab')


def experiment_tab():
    """
    Construct the initial experiment tab in the layout.
    """
    return html.Div([
        html.Div([
            html.Div([
                html.H6('Variable Selection', className='gs-header gs-table-header tiny-header'),
                dcc.Graph(id='variable-selection-graph', figure={'data': []})
            ], className="eight columns"),
            html.Div([
                html.H6('Variable Selection Method',
                        className='gs-header gs-table-header tiny-header'),
                dcc.Dropdown(
                    options=[
                        {'label': 'Lasso CV', 'value': 'LassoCV'},
                        {'label': 'Lasso CV + 1 std.', 'value': 'LassoCVStd'},
                        {'label': 'Lasso BIC', 'value': 'LassoBIC'},
                    ],
                    multi=True,
                    placeholder="Select a variable selection method",
                    id="variable-selection-methods"
                ),
                dcc.Dropdown(
                    options=SUBSET_METRICS,
                    multi=True,
                    placeholder="Select a variable selection metric",
                    id="variable-selection-metrics"
                )
            ], className="four columns")
        ], className="row"),
        html.Div([
            html.Div([
                html.H6('Prediction Error', className='gs-header gs-table-header tiny-header'),
                dcc.Graph(id='prediction-error-graph', figure={'data': []})
            ], className="eight columns"),
            html.Div([
                html.H6('Error Types', className='gs-header gs-table-header tiny-header'),
                dcc.Dropdown(
                    options=ERROR_TYPES,
                    multi=True,
                    placeholder="Select a prediction metric",
                    id="error-types"
                )
            ], className="four columns")
        ], className="row"),
        html.Div([
            html.Div([
                html.H6('# Trials', className='gs-header gs-table-header tiny-header'),
                dcc.Input(
                    id='num-trials',
                    placeholder='Enter a value...',
                    type='number',
                    inputmode='numeric',
                    min='1',
                    value='5'
                ),
            ], className="two columns"),
            html.Div([
                html.H6('# Samples Min', className='gs-header gs-table-header tiny-header'),
                dcc.Input(
                    id='samples-min',
                    placeholder='Enter a value...',
                    type='number',
                    inputmode='numeric',
                    min='5',
                    value='5'
                ),
            ], className="two columns"),
            html.Div([
                html.H6('# Samples Max', className='gs-header gs-table-header tiny-header'),
                dcc.Input(
                    id='samples-max',
                    placeholder='Enter a value...',
                    type='number',
                    inputmode='numeric',
                    min='0',
                    value='11'
                ),
            ], className="two columns"),
            html.Div([
                html.H6('# Samples Step', className='gs-header gs-table-header tiny-header'),
                dcc.Input(
                    id='samples-step',
                    placeholder='Enter a value...',
                    type='number',
                    inputmode='numeric',
                    min='0',
                    value='5'
                ),
            ], className="two columns"),
            html.Div([
                html.Button('Run Experiment', n_clicks=0,
                            id='run-button', className='button-primary')
            ], className='four columns', style={'float': 'right', 'vertical-align': 'bottom'})
        ], className='row'),
        # Hidden div inside the app that stores the experiment results dataframe in json
        html.Div(id='graph-data-df', style={'display': 'none'}),
    ], className='page', style={'display': 'none'}, id='experiment-tab')


DATA_TAB = 1
EXPERIMENT_TAB = 2

app.layout = \
    html.Div([
        dcc.Tabs(tabs=[
            {'label': 'Data', 'value': DATA_TAB},
            {'label': 'Experiment', 'value': EXPERIMENT_TAB}],
            value=1, vertical=False, id='tabs'),
        # Data tab
        data_tab(),
        # Experiment tab
        experiment_tab(),
    ], style={
        'width': '80%',
        'fontFamily': 'Sans-Serif',
        'margin-left': 'auto',
        'margin-right': 'auto'
    })


@app.callback(Output('data-tab', 'style'), [Input('tabs', 'value')])
def display_data_content(tabid):
    """
    Displays data tab when it is selected.
    """
    return {'display': 'block' if tabid == DATA_TAB else 'none'}


@app.callback(Output('experiment-tab', 'style'), [Input('tabs', 'value')])
def display_experiment_content(tabid):
    """
    Displays experiment tab when it is selected.
    """
    return {'display': 'block' if tabid == EXPERIMENT_TAB else 'none'}


from . import datatab
from . import experimenttab
