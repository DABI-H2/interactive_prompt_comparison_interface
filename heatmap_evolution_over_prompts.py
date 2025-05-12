import dash
from dash import dcc, html
import plotly.express as px
import numpy as np
import pandas as pd

# Dummy data
prompts = ['Prompt 1', 'Prompt 2', 'Prompt 3', 'Prompt 4']
metrics = ['Perplexity', 'BLEU Score', 'ROUGE Score']
data = np.array([
    [0.60, 0.55, 0.50],  # Normalized values
    [0.56, 0.58, 0.52],
    [0.52, 0.60, 0.54],
    [0.50, 0.65, 0.55]
])

df = pd.DataFrame(data, index=prompts, columns=metrics)

# Heatmap
fig = px.imshow(df, title='Heatmap of Metrics over Prompts')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
