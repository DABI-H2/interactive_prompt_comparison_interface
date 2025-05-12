import dash
from dash import dcc, html
import plotly.express as px
import numpy as np

# Dummy data
data = np.random.normal(loc=5, scale=2, size=1000)

# Histogram
fig = px.histogram(data, nbins=30, title='Distribution of Generated Text Lengths')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
