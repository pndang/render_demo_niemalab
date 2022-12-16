import datetime
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd 
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
import seaborn as sns
from PIL import Image, ImageDraw
import os 

# app = Dash(__name__)
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server 

# server.secret_key = os.environ.get('SECRET_KEY', 'my-secret-key')

app.config.suppress_callback_exceptions = True

# ------------------------------------------------------------------------------

# Function for calculating rolling averages
def calculate_rolling_avg(dataframe):
  dataframe['rolling_avg'] = 0
  for i in range(0, dataframe.shape[0]):
    idx = i
    last_idx = dataframe.shape[0]-1
    count = 0  # count number of values added (imaginary), to handle the first and last 3 rows
               # Idea: if index is 0, assume 3 "before" values already added, 
               #       if index is 1, assume 2 "before" values already added, 
               #       so on

    values_before = []
    while len(values_before) < 3 and count < 3:
      if i == 0:
        break
      count = 2 if idx == 1 else 1 if idx == 2 else 0
      values_before.append(dataframe.iloc[idx-1][5])
      idx -= 1
      count += 1   
    
    idx, count = i, 0
    values_after = []
    while len(values_after) < 3 and count < 3:
      if i == last_idx:
        break
      count = 2 if idx == last_idx-1 else 1 if i == last_idx-2 else 0
      values_after.append(dataframe.iloc[idx+1][5])
      idx += 1
      count += 1

    # Calculate the average
    average = np.mean(values_before + [dataframe.iloc[i][5]] + values_after)
    dataframe.at[i, 'rolling_avg'] = average

    # print(i)
    # print(values_before)
    # print(values_after)
    # print(values_before + [dataframe.iloc[i][4]] + values_after)
    # print(average)
    # print('***********')

  return None

# Importing and cleaning dataset
df = pd.read_csv('covid19variants.csv')

# ------------------------------------------------------------------------------

# WebApp layout with BOOTSTRAP theme stylesheet for component positioning
# Check 'covid19_variants_graphing_tool.py' for regular layout
app.layout = dbc.Container([
  dbc.Row([
    dbc.Col([html.Br(),
      html.H4("COVID-19 Variants Graphing App", style={'textAlign':'center'}),
      html.Br()], width=12)
  ]),

  dbc.Row([
    dbc.Col([
      dcc.DatePickerRange(
        id='my-date-picker-range',
        min_date_allowed=datetime.date(2021, 1, 1),
        max_date_allowed=datetime.date(2022, 9, 23),
        initial_visible_month=datetime.date(2021, 1, 1),
        end_date=datetime.date(2022, 9, 23)
    ), html.Br(), html.Br()], width=6),
    dbc.Col([
      dcc.Dropdown(
        id='mydropdown',
        options={x: x for x in df.variant_name.unique()},
        multi=True,
        placeholder='Select variants'
      )], width=6)
  ]),

  dbc.Row([
    dbc.Col([
      html.Div(id='output-date-picker-range'), html.Br()
    ], width=12),
    html.P("Choose color palette:"),
    dcc.RadioItems([
        {
            "label": html.Div(
                [
                    html.Img(src="/assets/inferno.png", height=30),
                    html.Div("inferno", style={'font-size': 15, 'padding-left': 1}),
                ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
            ),
            "value": "inferno",
        },
        {
            "label": html.Div(
                [
                    html.Img(src="/assets/icefire.png", height=30),
                    html.Div("icefire", style={'font-size': 15, 'padding-left': 1}),
                ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
            ),
            "value": "icefire",
        },
        {
            "label": html.Div(
                [
                    html.Img(src="/assets/rainbow.png", height=30),
                    html.Div("rainbow", style={'font-size': 15, 'padding-left': 1}),
                ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
            ),
            "value": "rainbow",
        },
        {
            "label": html.Div(
                [
                    html.Img(src="/assets/autumn.png", height=30),
                    html.Div("autumn", style={'font-size': 15, 'padding-left': 1}),
                ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
            ),
            "value": "autumn",
        },
        {
            "label": html.Div(
                [
                    html.Img(src="/assets/ocean.png", height=30),
                    html.Div("ocean", style={'font-size': 15, 'padding-left': 1}),
                ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
            ),
            "value": "ocean",
        },
    ], id='mypalette', value='mako')
  ]),

  html.Br(),

  dcc.Graph(id='my_plot')

])


@app.callback(
    [Output('output-date-picker-range', 'children'),
    Output('my_plot', 'figure')],
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('mydropdown', 'value'),
    Input('mypalette', 'value'),
    prevent_initial_call=True)


def update_output(start_date, end_date, dropdown_val, palette):
    string_prefix = 'You have selected: '
    plot_ready = False
    if start_date is not None:
        start_date_obj = datetime.date.fromisoformat(start_date)
        start_date_str = start_date_obj.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'Start Date: {} | '.format(start_date_str)
        plot_ready = True
    if end_date is not None:
        end_date_obj = datetime.date.fromisoformat(end_date)
        end_date_str = end_date_obj.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'End Date: {}'.format(end_date_str)
    if len(string_prefix) == len('You have selected: '):
        return 'Select a date to see it displayed here'
    if plot_ready and dropdown_val != None:
        data = df.loc[df['variant_name'].isin(dropdown_val)]
        date_col = 'date'
        variant_col = 'variant_name'

        # Check if time_col column is of type datetime, proceed if yes, convert 
        # to datetime if no
        if data.get(date_col).dtypes == '<M8[ns]':
            pass
        else:
            data['date'] = [datetime.datetime.strptime(date, '%Y-%m-%d').date()
                            for date in data[date_col]]
            date_col = 'date'

        data = data[(data[date_col] >= start_date_obj) & \
                    (data[date_col] <= end_date_obj)]

        print(start_date_obj)

        variants = data.get(variant_col).unique()

        # Creating a dictionary of variants (keys are variant names & values are 
        # subset dataframes of variants)
        variants_dict = {var: data[data[variant_col] == var] for var in variants}

        # Calculating rolling averages for each variant dataframes in dictionary 
        for value in variants_dict.values():
            value.reset_index(inplace=True)
            calculate_rolling_avg(value)  
            value.set_index('index', inplace=True)

        # Concatenating the dataframes into one for plotting, drop rows with N/A
        variants_wra = pd.concat(variants_dict.values())
        variants_wra = variants_wra.dropna(subset=[date_col])
        
        palette = sns.color_palette(palette, len(variants)).as_hex()

        fig = px.line(data_frame=variants_wra,
                      x=date_col,
                      y='rolling_avg',
                      color=variant_col,
                      color_discrete_sequence=palette)

        fig.update_layout(title='Daily Specimen Count by Variant, averaged 3 days prior/after',
                          yaxis_title='specimen count')

        return string_prefix, fig
    
    raise PreventUpdate



if __name__ == '__main__':
    app.run_server(debug=True)
    # app.run_server(debug=True, dev_tools_ui=None, dev_tools_props_check=None)

