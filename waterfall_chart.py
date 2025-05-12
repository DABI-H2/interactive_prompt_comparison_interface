import dash
from dash import dcc, html
import plotly.graph_objects as go

# Dummy data
categories = ['Perplexity', 'BLEU Score', 'ROUGE Score']
previous_values = [1.28, 0.6, 0.5]
current_values = [1.1, 0.65, 0.55]

# Calculate differences
differences = [current - previous for current, previous in zip(current_values, previous_values)]

# Waterfall chart
fig = go.Figure(go.Waterfall(
    x=categories,
    measure=["relative", "relative", "relative"],
    y=differences,
    decreasing={"marker": {"color": "Maroon"}},
    increasing={"marker": {"color": "Teal"}},
    totals={"marker": {"color": "deep sky blue"}}
))

fig.update_layout(title='Waterfall Chart of Differences between Previous and Current Outputs')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
