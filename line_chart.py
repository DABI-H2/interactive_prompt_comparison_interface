import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Dummy data
data = {
    'Prompt': ['Prompt 1', 'Prompt 2', 'Prompt 3', 'Prompt 4'],
    'Score': [0.7, 0.92, 0.81, 0.95]
}
df = pd.DataFrame(data)

# Line chart
fig = px.line(df, x='Prompt', y='Score', title='Score Over Prompts')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
