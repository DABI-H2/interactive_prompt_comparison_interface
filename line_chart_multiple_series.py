import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd

# Dummy data
data = {
    'Prompt': ['Prompt 1', 'Prompt 2', 'Prompt 3', 'Prompt 4'],
    'Perplexity': [0.60, 0.56, 0.52, 0.50],  # Normalized values
    'BLEU Score': [0.55, 0.58, 0.60, 0.65],
    'ROUGE Score': [0.50, 0.52, 0.54, 0.55]
}
df = pd.DataFrame(data)

# Line chart with multiple series
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Prompt'], y=df['Perplexity'], mode='lines+markers', name='Perplexity'))
fig.add_trace(go.Scatter(x=df['Prompt'], y=df['BLEU Score'], mode='lines+markers', name='BLEU Score'))
fig.add_trace(go.Scatter(x=df['Prompt'], y=df['ROUGE Score'], mode='lines+markers', name='ROUGE Score'))

fig.update_layout(title='Development of Results over Prompts', xaxis_title='Prompt', yaxis_title='Score')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
