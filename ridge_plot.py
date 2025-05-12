import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Dummy data
data = {
    'Prompt': ['Prompt 1'] * 3 + ['Prompt 2'] * 3 + ['Prompt 3'] * 3 + ['Prompt 4'] * 3,
    'Metric': ['Perplexity', 'BLEU Score', 'ROUGE Score'] * 4,
    'Value': [0.60, 0.55, 0.50, 0.56, 0.58, 0.52, 0.52, 0.60, 0.54, 0.50, 0.65, 0.55]  # Normalized values
}
df = pd.DataFrame(data)

# Ridge plot
fig = px.violin(df, x='Value', y='Prompt', color='Metric', 
                title='Ridge Plot of Metrics over Prompts', 
                box=True, points='all', 
                hover_data=df.columns)

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
