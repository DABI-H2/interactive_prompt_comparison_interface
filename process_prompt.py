import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sacrebleu
from rouge import Rouge
import numpy as np
import dash_ag_grid as dag


# Dummy data for ground truths
ground_truth_data = {
    "Describe the weather": {"output": "The weather is pleasant today.", "confidence": 0.95, "response_time": 0.9, "bleu": 0.85, "rouge": 0.88},
    "Describe the current weather": {"output": "It is raining outside.", "confidence": 0.78, "response_time": 0.7, "bleu": 0.75, "rouge": 0.80},
    "What is the weather like?": {"output": "The sun is shining.", "confidence": 0.90, "response_time": 0.6, "bleu": 0.40, "rouge": 0.82},
    "How is the weather today?": {"output": "Today is a cloudy day.", "confidence": 0.85, "response_time": 0.8, "bleu": 0.78, "rouge": 0.79},
    "What's the weather forecast?": {"output": "The forecast predicts rain in the afternoon.", "confidence": 0.80, "response_time": 0.65, "bleu": 0.77, "rouge": 0.81},
    "Is it sunny outside?": {"output": "Yes, it is quite sunny outside.", "confidence": 0.68, "response_time": 0.5, "bleu": 0.92, "rouge": 0.84},
    "Will it rain today?": {"output": "There is a high chance of rain today.", "confidence": 0.82, "response_time": 0.75, "bleu": 0.79, "rouge": 0.83},
    "Tell me about the weather.": {"output": "The weather is mild and partly cloudy.", "confidence": 0.87, "response_time": 0.7, "bleu": 0.81, "rouge": 0.85},
    "How's the weather now?": {"output": "Currently, it is foggy with low visibility.", "confidence": 0.77, "response_time": 0.55, "bleu": 0.76, "rouge": 0.78},
    "What does the weather look like?": {"output": "The sky is overcast and it looks like it might rain.", "confidence": 0.80, "response_time": 0.7, "bleu": 0.79, "rouge": 0.82},
    "Is it raining?": {"output": "Yes, it is raining right now.", "confidence": 0.40, "response_time": 0.5, "bleu": 0.45, "rouge": 0.88},
    "Is there a storm coming?": {"output": "There is a storm warning for the evening.", "confidence": 0.83, "response_time": 0.8, "bleu": 0.48, "rouge": 0.81},
    "Will there be snow today?": {"output": "Snow is expected later today.", "confidence": 0.72, "response_time": 0.9, "bleu": 0.70, "rouge": 0.75},
    "Is it hot outside?": {"output": "The temperature is quite high today.", "confidence": 0.88, "response_time": 0.55, "bleu": 0.63, "rouge": 0.87},
    "What is the humidity level?": {"output": "The humidity level is around 60%.", "confidence": 0.76, "response_time": 0.8, "bleu": 0.74, "rouge": 0.57},
    "Is it windy today?": {"output": "Yes, there are strong winds today.", "confidence": 0.85, "response_time": 0.6, "bleu": 0.79, "rouge": 0.91},
    "How cold is it outside?": {"output": "It is quite cold outside.", "confidence": 0.80, "response_time": 0.75, "bleu": 0.78, "rouge": 0.82},
    "What's the UV index today?": {"output": "The UV index is high, around 8.", "confidence": 0.84, "response_time": 0.65, "bleu": 0.50, "rouge": 0.83},
    "Is it a good day for a walk?": {"output": "Yes, it's a good day for a walk.", "confidence": 0.90, "response_time": 0.55, "bleu": 0.85, "rouge": 0.98},
    "Are there any weather warnings?": {"output": "There is a warning for heavy rainfall.", "confidence": 0.59, "response_time": 0.8, "bleu": 0.77, "rouge": 0.80}
}
previous_metrics = []
prompt_contents = {}

# Function to simulate processing the prompt and generating output
def process_prompt(prompt):
    if prompt in ground_truth_data:
        return ground_truth_data[prompt]["output"]
    else:
        return "No relevant data."

# Function to calculate metrics
def calculate_metrics(output, ground_truth_key):
    ground_truth = ground_truth_data[ground_truth_key]
    bleu_score = ground_truth["bleu"]
    rouge_score = ground_truth["rouge"]
    confidence = ground_truth.get("confidence", 0)
    response_time = ground_truth.get("response_time", 0)
    return bleu_score, rouge_score, confidence, response_time

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Prompt Interface to Metrics", style={'text-align': 'center'}),
    html.Div([
        dcc.Input(id="input-prompt", type="text", placeholder="Enter your prompt here", style={'width': '60%'}),
        html.Button('Submit', id='submit-button', n_clicks=0)
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.Div(id="output-container", style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.Div([
        html.Label('Select Prompt:', style={'font-weight': 'bold'}),
        dcc.Dropdown(id="prompt-dropdown", options=[], multi=False, placeholder="Select a prompt", style={'width': '90%'}),
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
    dag.AgGrid(
        id='legend-container',
        columnDefs=[
            {"headerName": "Prompt", "field": "prompt"},
            {"headerName": "Content", "field": "content"},
            {"headerName": "Show", "field": "show", "checkboxSelection": True}
        ],
        rowData=[],  # This will be filled dynamically
        defaultColDef={"flex": 1, "sortable": True, "filter": True, "resizable": True},
        style={'height': '200px', 'width': '100%', 'margin-top': '20px'},
        dashGridOptions={
            "rowSelection": "multiple"
        },
    ),
    html.Div([
        html.Div([
            html.Label('Select metrics to display on radar plot:', style={'font-weight': 'bold'}),
            dcc.Checklist(
                id='radar-metrics',
                options=[
                    {'label': 'BLEU Score', 'value': 'BLEU Score'},
                    {'label': 'ROUGE Score', 'value': 'ROUGE Score'},
                    {'label': 'Confidence', 'value': 'Confidence'},
                    {'label': 'Response Time', 'value': 'Response Time'},
                ],
                value=['BLEU Score', 'ROUGE Score', 'Confidence', 'Response Time'],
                style={'border': '1px solid black', 'padding': '5px', 'background-color': '#f9f9f9', 'font-size': '12px'}
            ),
            dcc.Graph(id="metrics-radar-plot")
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label('Select metrics to display on heatmap:', style={'font-weight': 'bold'}),
            dcc.Checklist(
                id='heatmap-metrics',
                options=[
                    {'label': 'BLEU Score', 'value': 'BLEU Score'},
                    {'label': 'ROUGE Score', 'value': 'ROUGE Score'},
                    {'label': 'Confidence', 'value': 'Confidence'},
                    {'label': 'Response Time', 'value': 'Response Time'},
                ],
                value=['BLEU Score', 'ROUGE Score', 'Confidence', 'Response Time'],
                style={'border': '1px solid black', 'padding': '5px', 'background-color': '#f9f9f9', 'font-size': '12px'}
            ),
            dcc.Graph(id="metrics-heatmap")
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'})
    ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            html.Label('Select metrics to display on pair plot:', style={'font-weight': 'bold'}),
            dcc.Checklist(
                id='pair-metrics',
                options=[
                    {'label': 'Parameter 1', 'value': 'Parameter 1'},
                    {'label': 'Parameter 2', 'value': 'Parameter 2'},
                    {'label': 'BLEU Score', 'value': 'BLEU Score'},
                    {'label': 'ROUGE Score', 'value': 'ROUGE Score'}
                ],
                value=['Parameter 1', 'Parameter 2', 'BLEU Score', 'ROUGE Score'],
                style={'border': '1px solid black', 'padding': '5px', 'background-color': '#f9f9f9', 'font-size': '12px'}
            ),
            dcc.Graph(id="metrics-pair-plot")
        ], style={'width': '98%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            dcc.Graph(id="metrics-cdf-plot-length")
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id="metrics-cdf-plot-confidence")
        ], style={'width': '48%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    html.Div([
        html.Div([
            dcc.Graph(id="metrics-cdf-plot-response-time")
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justify-content': 'space-between'})
])


@app.callback(
    [Output("output-container", "children"),
     Output("metrics-radar-plot", "figure"),
     Output("metrics-heatmap", "figure"),
     Output("metrics-pair-plot", "figure"),
     Output("metrics-cdf-plot-length", "figure"),
     Output("metrics-cdf-plot-confidence", "figure"),
     Output("metrics-cdf-plot-response-time", "figure"),
     Output("legend-container", "rowData")],
    [Input("submit-button", "n_clicks"),
     Input("radar-metrics", "value"),
     Input("pair-metrics", "value"),
     Input("legend-container", "selectedRows")],
    [State("input-prompt", "value"),
     State("legend-container", "rowData")]
)
def update_output(n_clicks, radar_metrics, pair_metrics, selected_rows, value, current_row_data):
    global previous_metrics
    global prompt_contents
    
    ctx = dash.callback_context
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

    output = ""
    
    if n_clicks > 0 and value and triggered_input == "submit-button":
        output = process_prompt(value)
        
        if value in ground_truth_data:
            bleu_score, rouge_score, confidence, response_time = calculate_metrics(output, value)
    
            current_metrics = {
                'Metric': ['BLEU Score', 'ROUGE Score', 'Confidence', 'Response Time'],
                'Value': [bleu_score, rouge_score, confidence, response_time],
                'Prompt': f'Prompt {n_clicks}'  # Add an identifier for the prompt
            }
            previous_metrics.append(current_metrics)
            prompt_contents[f'Prompt {n_clicks}'] = value

    # If the legend was updated, use the selected prompts
    if selected_rows:
        selected_prompt_keys = [row['prompt'] for row in selected_rows]
    else:
        selected_prompt_keys = list(prompt_contents.keys())

    radar_fig = go.Figure()
    if radar_metrics:
        for metrics in previous_metrics:
            if metrics['Prompt'] in selected_prompt_keys:
                filtered_values = [metrics['Value'][metrics['Metric'].index(metric)] for metric in radar_metrics]
                filtered_metrics = [metric for metric in radar_metrics]
                radar_fig.add_trace(go.Scatterpolar(
                    r=filtered_values,
                    theta=filtered_metrics,
                    fill='toself',
                    name=metrics['Prompt']
                ))

    radar_fig.update_layout(title='Radar Plot of Metrics')

    # Prepare data for the heatmap
    heatmap_data = {metrics['Prompt']: [metrics['Value'][metrics['Metric'].index(metric)] for metric in radar_metrics] for metrics in previous_metrics if metrics['Prompt'] in selected_prompt_keys}
    heatmap_df = pd.DataFrame(heatmap_data, index=radar_metrics)
    heatmap_fig = px.imshow(heatmap_df, text_auto=True, title='Heatmap of Metrics')

    pair_plot_data = {
        'Parameter 1': np.random.rand(10),
        'Parameter 2': np.random.rand(10),
        'BLEU Score': np.random.rand(10),
        'ROUGE Score': np.random.rand(10)
    }
    pair_plot_df = pd.DataFrame(pair_plot_data)
    if pair_metrics:
        pair_plot_fig = px.scatter_matrix(pair_plot_df, dimensions=pair_metrics, title='Pair Plot of Parameters and Metrics')
        pair_plot_fig.update_layout(height=800)  # Increase height for more vertical spacing

    cdf_data_length = np.random.poisson(5, 100)
    cdf_fig_length = go.Figure()
    cdf_fig_length.add_trace(go.Scatter(
        x=sorted(cdf_data_length),
        y=np.arange(1, len(cdf_data_length) + 1) / len(cdf_data_length),
        mode='lines',
        name='Length'
    ))
    cdf_fig_length.update_layout(title='CDF of Generated Text Lengths')

    cdf_data_confidence = np.random.uniform(0.8, 1.0, 100)
    cdf_fig_confidence = go.Figure()
    cdf_fig_confidence.add_trace(go.Scatter(
        x=sorted(cdf_data_confidence),
        y=np.arange(1, len(cdf_data_confidence) + 1) / len(cdf_data_confidence),
        mode='lines',
        name='Confidence'
    ))
    cdf_fig_confidence.update_layout(title='CDF of Model Confidence Scores')

    cdf_data_response_time = np.random.uniform(0.5, 1.0, 100)
    cdf_fig_response_time = go.Figure()
    cdf_fig_response_time.add_trace(go.Scatter(
        x=sorted(cdf_data_response_time),
        y=np.arange(1, len(cdf_data_response_time) + 1) / len(cdf_data_response_time),
        mode='lines',
        name='Response Time'
    ))
    cdf_fig_response_time.update_layout(title='CDF of Response Times')

    legend_items = [
        {"prompt": prompt, "content": content, "show": True}
        for prompt, content in prompt_contents.items()
    ]

    return f"Generated Output: {output}", radar_fig, heatmap_fig, pair_plot_fig, cdf_fig_length, cdf_fig_confidence, cdf_fig_response_time, legend_items



if __name__ == '__main__':
    app.run_server(debug=True)