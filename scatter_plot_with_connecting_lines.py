import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd

# Dummy data for multiple prompts
data = {
    'Prompt': ['Prompt 1', 'Prompt 2', 'Prompt 3', 'Prompt 4'],
    'Perplexity': [1.28, 1.20, 1.15, 1.10],
    'BLEU Score': [0.55, 0.60, 0.63, 0.65],
    'ROUGE Score': [0.50, 0.52, 0.53, 0.55]
}
df = pd.DataFrame(data)

# Scatter plot with connecting lines
fig = go.Figure()

for metric in df.columns[1:]:  # Skip the 'Prompt' column
    fig.add_trace(go.Scatter(x=df['Prompt'], 
                             y=df[metric], 
                             mode='lines+markers', 
                             name=metric))

fig.update_layout(title='Comparison of Metrics across Prompts',
                  xaxis_title='Prompt', yaxis_title='Value')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
