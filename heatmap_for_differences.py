import dash
from dash import dcc, html
import plotly.express as px
import numpy as np
import pandas as pd

# Dummy data
metrics = ['BLEU Score', 'ROUGE-1', 'ROUGE-L']
differences = np.array([[0.05, -0.03, 0.02],
                        [-0.01, 0.04, -0.02],
                        [0.03, 0.01, 0.05]])

df = pd.DataFrame(differences, columns=metrics, index=metrics)

# Heatmap
fig = px.imshow(df, title='Heatmap of Differences between Previous and Current Outputs')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
