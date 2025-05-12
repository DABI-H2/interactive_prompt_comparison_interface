import dash
from dash import dcc, html
import plotly.express as px
import numpy as np
import pandas as pd

# Dummy data
prompt_length = np.random.randint(1, 20, size=100)  # X axis: Prompt Length
output_relevance = np.random.randint(1, 10, size=100)  # Y axis: Output Relevance
frequency = np.random.randint(1, 50, size=100)  # Z axis: Frequency of Occurrence

df = pd.DataFrame({'Prompt Length': prompt_length, 'Output Relevance': output_relevance, 'Frequency': frequency})

# Heatmap
fig = px.density_heatmap(df, x='Prompt Length', y='Output Relevance', z='Frequency', nbinsx=20, nbinsy=10, title='Heatmap of Prompt Length vs. Output Relevance')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
