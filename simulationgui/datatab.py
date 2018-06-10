import io
import base64
from dash.dependencies import Input, Output, State, Event
import pandas as pd
import plotly.graph_objs as go

from .server import app

NEW_ROW = {"Name": "Name", "Type": "Normal", "Mean": "0.0", "Variance": "0.0"}


@app.callback(Output('input-data-df', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename')])
def save_input_data(contents, filename):
    if contents is None:
        return ""
    _, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    if filename is None:
        return ""
    if filename.endswith(".csv"):
        print("Saving CSV")
        return pd.DataFrame.from_csv(io.StringIO(decoded.decode('utf-8'))).to_json()


@app.callback(Output('distributions-table', 'rows'),
              [Input('input-data-df', 'children'),
               Input('upload-data-button', 'n_clicks_timestamp'),
               Input('add-variable-button', 'n_clicks_timestamp')],
              state=[State('distributions-table', 'rows')])
def add_row_distributions_table(uploaded_data, upload_timestamp, add_var_timestamp, distributions_dict):
    """
    Adds a row to the distribution table when the add variable button is clicked.
    Also updates the distribution variables when files are uploaded.
    """
    distributions_table = pd.DataFrame(distributions_dict)
    if upload_timestamp is None and add_var_timestamp is None:
        return distributions_table.to_dict('records')
    # If the add variable button was clicked
    if (upload_timestamp is None and not add_var_timestamp is None) or \
            (not None in [upload_timestamp, add_var_timestamp] and add_var_timestamp > upload_timestamp):
        return distributions_table.append(NEW_ROW, ignore_index=True).to_dict('records')
    # otherwise, data was uploaded
    if uploaded_data is None or uploaded_data == "":
        return distributions_dict
    # update table
    uploaded_df = pd.read_json(uploaded_data)
    for newcol in uploaded_df.columns:
        current_vars = distributions_table['Name'].tolist()
        if not 'Name' in distributions_table.columns or not newcol in current_vars:
            row = NEW_ROW.copy()
            row['Name'] = newcol
            distributions_table = distributions_table.append(row, ignore_index=True)
    return distributions_table.to_dict('records')


@app.callback(Output('correlation-table', 'rows'),
              [Input('distributions-table-df', 'children')],
              state=[State('correlation-table', 'rows')])
def update_correlation_table(dist_table_json, corr_table_dict):
    """
    Updates the correlation table whenever the distributions table is updated.
    """
    dist_table = pd.read_json(dist_table_json)
    dist_table.index = dist_table.index.map(int)
    dist_table = dist_table.sort_index()
    if not 'Name' in dist_table.columns:
        return [{}]

    corr_table = pd.DataFrame(corr_table_dict)
    names = dist_table['Name'].tolist()
    names = list(filter(lambda x: x != 'Name', names))

    for col in corr_table.columns:
        if col == 'Name':
            continue
        if not col in names:
            # delete col and row
            corr_table = corr_table.drop([col], axis=1)
            corr_table = corr_table.query("Name != '{0}'".format(col))
    for name in names:
        if not name in corr_table.columns:
            # add col and row
            row = {}
            for col in corr_table.columns:
                row[col] = 0.0
            row["Name"] = name
            corr_table = corr_table.append(row, ignore_index=True)
            corr_table[name] = 0.0

    # remove rows without names
    corr_table = corr_table.dropna()
    # sort rows
    if names:
        sorted_cols = sorted(corr_table.index, key=lambda ind: names.index(
            corr_table.at[ind, 'Name']))
        corr_table = corr_table.reindex_axis(sorted_cols, axis=0)

    return corr_table.to_dict('records')


@app.callback(Output('correlation-table', 'columns'),
              [Input('correlation-table', 'rows')],
              state=[State('distributions-table-df', 'children')])
def update_corr_table_columns(rows, dist_table_json):
    """
    Updates the columns of the correlation table whenever the rows are updated.
    """
    corr_table = pd.DataFrame(rows)
    dist_table = pd.read_json(dist_table_json)
    dist_table.index = dist_table.index.map(int)
    dist_table = dist_table.sort_index()

    if not 'Name' in dist_table.columns:
        return corr_table.columns
    names = dist_table['Name'].tolist()
    names = list(filter(lambda x: x != 'Name', names))

    # sort columns based on position in distribution table, where the Name column goes first always
    sorted_cols = sorted(corr_table.columns, key=lambda col:
                         -1 if col == "Name" else names.index(col))
    corr_table = corr_table.reindex_axis(sorted_cols, axis=1)

    return corr_table.columns


@app.callback(Output('distributions-table-df', 'children'),
              [Input('distributions-table', 'row_update'),
               Input('distributions-table', 'rows'),
               Input('add-variable-button', 'n_clicks')],
              state=[State('distributions-table-df', 'children')],
              events=[Event('add-variable-button', 'click')])
def update_distributions_df(update, rows, nclicks, df_json):
    """
    Updates the internal representation of the distributions table when the
    add variable button is clicked.
    """
    newrows = pd.DataFrame(rows)
    # Add Variable button was clicked
    if df_json == newrows.to_json():
        newrows = newrows.append(NEW_ROW, ignore_index=True)
    return newrows.to_json()


@app.callback(Output('correlation-heatmap', 'figure'),
              [Input('correlation-table', 'columns')],
              state=[State('correlation-table', 'rows')])
def update_correlation_heatmap(cols, corr_dict):
    """
    Updates the correlation heatmap when the correlation table is updated.
    """
    corr_df = pd.DataFrame(corr_dict)
    corr_df = corr_df[cols]
    if not 'Name' in corr_df.columns:
        return {'data': [{'type': 'heatmap'}]}
    # flip y axis so it shows up correctly
    corr_df = corr_df.reindex(index=corr_df.index[::-1])

    data = {
        'z': corr_df.drop(['Name'], axis=1).values,
        'x': corr_df.drop(['Name'], axis=1).columns,
        'y': corr_df['Name'].tolist(),
        'colorscale': 'Viridis',
        'type': 'heatmap'
    }
    layout = go.Layout(
        title="Correlation heatmap",
        xaxis={'side': 'top'},
        yaxis={},
    )
    return {'data': [data], 'layout': layout}
