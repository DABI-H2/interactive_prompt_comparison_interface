import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Dummy data
data = {
    'Metric': ['Perplexity', 'BLEU Score', 'ROUGE Score'],
    'Difference': [0.18, 0.05, 0.05]  # Current - Previous
}
df = pd.DataFrame(data)

# Difference bar chart
fig = px.bar(df, x='Metric', y='Difference', title='Difference between Previous and Current Outputs')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
