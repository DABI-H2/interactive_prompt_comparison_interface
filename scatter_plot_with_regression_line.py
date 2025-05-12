import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Dummy data
data = {
    'Parameter': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'BLEU Score': [0.45, 0.50, 0.55, 0.60, 0.62, 0.63, 0.65, 0.66, 0.67, 0.68]
}
df = pd.DataFrame(data)

# Scatter plot with regression line
fig = px.scatter(df, x='Parameter', y='BLEU Score', trendline='ols', title='Parameter vs BLEU Score')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
