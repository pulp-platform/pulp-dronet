#-----------------------------------------------------------------------------#
# Copyright(C) 2024 University of Bologna, Italy, ETH Zurich, Switzerland.    #
# All rights reserved.                                                        #
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License");             #
# you may not use this file except in compliance with the License.            #
# See LICENSE.apache.md in the top directory for details.                     #
# You may obtain a copy of the License at                                     #
#                                                                             #
#   http://www.apache.org/licenses/LICENSE-2.0                                #
#                                                                             #
# Unless required by applicable law or agreed to in writing, software         #
# distributed under the License is distributed on an "AS IS" BASIS,           #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
#                                                                             #
# File:    logger_app.py                                                      #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniel Rieben		    <riebend@student.ethz.ch>                 #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

from ds_logger import DatasetLogger
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import logging
import sys
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Get a handle to the crazyflie
uri = 'radio://0/80/2M/E7E7E7E7E7'
ds_logger = DatasetLogger(uri)
ds_logger.start_logging()

log_configs = []
for group_name in ds_logger.log_values:
    for val_name in ds_logger.log_values[group_name].keys():
        log_configs.append({
            "label": "{}.{}".format(group_name, val_name),
            "value": "{}.{}".format(group_name, val_name)
        })

# Initialize the logger app interface
logger_app = dash.Dash(title="DS collector", update_title=None)
# Define the layout of the logger app
logger_app.layout = html.Div(children=[
    html.Div([
        html.H1(children='Log values'),
        dcc.Graph(id='Data-Graph', animate=True),
        dcc.Interval(id='interval-component', interval=1000),
    ]),
    html.Div([
        dcc.Dropdown(
            id='log-values-dropdown',
            options=log_configs,
            multi=True
        )
    ]),
    html.Div(id='hidden-div', style={'display': 'none'})
])

selected_values = []
@logger_app.callback(Output('Data-Graph', 'figure'), Input('interval-component', 'n_intervals'), Input('log-values-dropdown', 'value'))
def update_graph_scatter(n_intervals, drowdown_value):
    global selected_values
    ctx = dash.callback_context
    component_id = ctx.triggered[0]["prop_id"].split('.')[0]

    if component_id == "log-values-dropdown":
        selected_values = drowdown_value

    if len(selected_values) > 0:
        fig = make_subplots(rows=len(selected_values), cols=1)
        for row, name in enumerate(selected_values):
            group_name, val_name = name.split('.')
            if len(ds_logger.log_values[group_name]["timestamp"]) > 0:
                # print("{}.{}".format(group_name, val_name))
                t = [x / 1.0e6 for x in ds_logger.log_values[group_name]["timestamp"]]
                y = ds_logger.log_values[group_name][val_name]
                fig.add_trace(
                    go.Scatter(x=t,
                               y=y,
                               name=val_name,
                               mode='lines+markers'),
                    row=row + 1, col=1
                )
                fig.update_xaxes(range=[min(t), max(t)], row=row+1, col=1)

    else:
        fig = {'data': []}

    return fig


logger_app.run_server(debug=False)

