import dash
from dash import dcc, html
import plotly.graph_objects as go

# Dummy data
categories = ['BLEU Score', 'ROUGE-1', 'ROUGE-L', 'Accuracy', 'F1 Score']
previous_values = [0.7, 0.6, 0.69, 0.8, 0.75]
current_values = [0.55, 0.79, 0.48, 0.82, 0.98]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=previous_values, theta=categories, fill='toself', name='Previous'))
fig.add_trace(go.Scatterpolar(r=current_values, theta=categories, fill='toself', name='Current'))

fig.update_layout(title='Radar Chart of Previous vs Current Outputs')

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
