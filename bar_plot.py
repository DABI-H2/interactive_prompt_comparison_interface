import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Dummy data
data = {
    'Setting': ['Setting 1', 'Setting 2', 'Setting 3'],
    'Frequency': [10, 15, 7]
}
df = pd.DataFrame(data)

# Bar chart
fig = px.bar(df, x='Setting', y='Frequency', title='Frequency of Settings')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
