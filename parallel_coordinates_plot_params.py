import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Dummy data
data = {
    'Parameter 1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Parameter 2': [5, 3, 6, 2, 7, 1, 8, 0, 9, 4],
    'BLEU Score': [0.45, 0.50, 0.55, 0.60, 0.62, 0.63, 0.65, 0.66, 0.67, 0.68],
    'ROUGE Score': [0.40, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
}
df = pd.DataFrame(data)

# Parallel coordinates plot
fig = px.parallel_coordinates(df, color='BLEU Score', 
                              dimensions=['Parameter 1', 'Parameter 2', 'BLEU Score', 'ROUGE Score'],
                              title='Parallel Coordinates Plot of Parameters and Metrics')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
