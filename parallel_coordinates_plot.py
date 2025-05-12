import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Dummy data
data = {
    'Prompt': ['Prompt 1', 'Prompt 2', 'Prompt 3', 'Prompt 4'],
    'Perplexity': [0.60, 0.56, 0.52, 0.50],  # Normalized values
    'BLEU Score': [0.55, 0.58, 0.60, 0.65],
    'ROUGE Score': [0.50, 0.52, 0.54, 0.55]
}
df = pd.DataFrame(data)

# Parallel coordinates plot
fig = px.parallel_coordinates(df, color='Perplexity', 
                              dimensions=['Perplexity', 'BLEU Score', 'ROUGE Score'],
                              title='Parallel Coordinates Plot of Metrics over Prompts')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
