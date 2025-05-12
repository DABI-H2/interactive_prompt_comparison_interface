import dash
from dash import dcc, html
import plotly.graph_objects as go

# Define nodes and their labels
nodes = [
    'User', 
    'Input Prompt', 
    'Prompt History', 
    'Adjust Parameters', 
    'Generate Output', 
    'Output Visualization', 
    'Compare Results'
]

# Define the links
links = [
    {'source': 0, 'target': 1, 'value': 1},  # User -> Input Prompt
    {'source': 1, 'target': 2, 'value': 1},  # Input Prompt -> Prompt History
    {'source': 1, 'target': 3, 'value': 1},  # Input Prompt -> Adjust Parameters
    {'source': 3, 'target': 4, 'value': 1},  # Adjust Parameters -> Generate Output
    {'source': 4, 'target': 5, 'value': 1},  # Generate Output -> Output Visualization
    {'source': 5, 'target': 6, 'value': 1},  # Output Visualization -> Compare Results
    {'source': 6, 'target': 3, 'value': 1},  # Compare Results -> Adjust Parameters
    {'source': 2, 'target': 6, 'value': 1},  # Prompt History -> Compare Results
]

# Create the Sankey diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes,
        color="blue"
    ),
    link=dict(
        source=[link['source'] for link in links],
        target=[link['target'] for link in links],
        value=[link['value'] for link in links],
        color="gray"
    )
))

# Update layout
fig.update_layout(
    title_text="Directed Interaction Diagram of the Prompt Engineering System",
    font_size=10,
    margin=dict(l=10, r=10, t=50, b=10),
    height=600,  # Adjust height for better spacing
)

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
