import dash
from dash import dcc, html
import plotly.figure_factory as ff
import numpy as np

# Dummy data
data = np.random.normal(loc=5, scale=2, size=1000)

# CDF plot
fig = ff.create_distplot([data], group_labels=['Data'], curve_type='normal', show_hist=False, show_rug=False)
fig.update_layout(title='CDF of Generated Data')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
