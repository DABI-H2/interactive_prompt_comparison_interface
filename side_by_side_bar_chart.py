import dash
from dash import dcc, html
import plotly.graph_objects as go

# Dummy data
categories = ['Perplexity', 'BLEU Score', 'ROUGE Score']
previous_values = [1.28, 0.6, 0.5]
current_values = [1.1, 0.65, 0.55]

# Side-by-side bar chart
fig = go.Figure(data=[
    go.Bar(name='Previous', x=categories, y=previous_values),
    go.Bar(name='Current', x=categories, y=current_values)
])
fig.update_layout(barmode='group', title='Comparison of Previous and Current Outputs')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
