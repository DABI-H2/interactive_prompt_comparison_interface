squad_v2_path = r"C:\Users\desim\Documents\UvA\MMA\SQUAD\dev-v2.0.json"
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as colors
import pandas as pd
import sacrebleu
from rouge import Rouge
import numpy as np
import json
from transformers import pipeline
import dash_ag_grid as dag
import evaluate
import time
from umap import UMAP
import clip
import torch
import random
from sentence_transformers import SentenceTransformer


# Set the random seed for reproducibility
seed_value = 42
def set_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)  # Ensure single-threaded operations


color_palette = colors.qualitative.Plotly  # You can choose other palettes like D3, G10, etc.


# Load the Sentence Transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_model.eval()

def calculate_sentence_embedding(input_text):
    set_seed(seed_value=seed_value)
    return sentence_model.encode([input_text])


def calculate_clip(input_text):
    return calculate_sentence_embedding(input_text)
    '''
    model, preprocess = clip.load("ViT-B/32", device='cuda')
    text = clip.tokenize(input_text).to('cuda')
    with torch.no_grad():
        return model.encode_text(text).cpu()'''

def calculate_umap(embeddings):
    umap_embeddings = UMAP(metric='cosine', n_components=2).fit_transform(embeddings)
    umap_x, umap_y = umap_embeddings[:, 0], umap_embeddings[:, 1]
    return umap_x, umap_y

# Load SQuAD dataset
with open(squad_v2_path) as f:
    squad_data = json.load(f)

#Load evaluation metrics
squad_metric = evaluate.load('squad_v2')

# Preprocess SQuAD data to create ground truth entries
ground_truth_data = {}
offset = 20
n_articles = 3
n_paragraphs = 1
n_qas = 1
for article in squad_data['data'][offset:offset+n_articles]:
    for paragraph in article['paragraphs'][:n_paragraphs]:
        context = paragraph['context']
        for qa in paragraph['qas'][:n_qas]:
            question = qa['question']
            if qa['is_impossible']:
                answer_text = ""
            else:
                answer_text = qa['answers'][0]['text']
            ground_truth_data[question] = {
                "output": answer_text,
                "context": context,
                #"confidence": np.random.uniform(0.7, 1.0),
                #"response_time": np.random.uniform(0.5, 1.5),
                #"exact_match": np.random.uniform(0.6, 1.0),
                #"f1": np.random.uniform(0.6, 1.0)
            }


previous_metrics = []
prompt_contents = {}

# Load a pre-trained QA model
qa_pipeline = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")

# Function to simulate processing the prompt and generating output
def process_prompt(prompt):
    outputs = {}
    for question, data in ground_truth_data.items():
        context = data["context"]
        full_prompt = f"{prompt}: {question}"
        start_time = time.time()
        result = qa_pipeline(question=full_prompt, context=context)
        end_time = time.time()
        response_time = end_time - start_time
        outputs[question] = {
            'answer': result['answer'],
            'response_time': response_time,
            'confidence': result['score']  # Using model's confidence score
        }
    return outputs


# Function to calculate metrics
def calculate_metrics(outputs):
    total_exact_match = 0
    total_f1_score = 0
    total_confidence = 0
    total_response_time = 0
    num_questions = len(outputs)

    predictions = []
    references = []

    for question, output in outputs.items():
        ground_truth = ground_truth_data[question]
        predictions.append({
            'id': question,
            'prediction_text': output['answer'],
            'no_answer_probability': 0.0  # Assuming a zero probability for no answer
        })
        references.append({
            'id': question,
            'answers': {
                'text': [ground_truth["output"]],
                'answer_start': [0]  # Dummy value; start positions are not used in evaluation
            }
        })

        results = squad_metric.compute(predictions=predictions, references=references)
        exact_match = results.get('exact', 0)/100
        f1_score = results.get('f1', 0)/100
        confidence = output['confidence']
        response_time = output['response_time']

        total_exact_match += exact_match
        total_f1_score += f1_score
        total_confidence += confidence
        total_response_time += response_time

        # Clear the lists for the next iteration
        predictions.clear()
        references.clear()

    avg_exact_match = total_exact_match / num_questions
    avg_f1_score = total_f1_score / num_questions
    avg_confidence = total_confidence / num_questions
    avg_response_time = total_response_time / num_questions

    return {
        "exact_match": avg_exact_match,
        "f1": avg_f1_score,
        "confidence": avg_confidence,
        "response_time": avg_response_time
    }

all_outputs = []

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Prompt Interface to Metrics", style={'text-align': 'center'}),
    html.Div([
        dcc.Input(id="input-prompt", type="text", placeholder="Enter your prompt here", style={'width': '60%'}),
        html.Button('Submit', id='submit-button', n_clicks=0)
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.Div(id="output-container", style={'text-align': 'center', 'margin-bottom': '20px'}),
    dag.AgGrid(
        id='legend-container',
        columnDefs=[
            {"headerName": "Prompt", "field": "prompt"},
            {"headerName": "Content", "field": "content"},
            {"headerName": "Show", "field": "show", "checkboxSelection": True}
        ],
        rowData=[],
        defaultColDef={"flex": 1, "sortable": True, "filter": True, "resizable": True},
        style={'height': '200px', 'width': '100%', 'margin-top': '20px'},
        dashGridOptions={"rowSelection": "multiple"},
    ),
    dcc.Tab(label='Scatter Plot', children=[
            dcc.Graph(id="scatter-plot")
        ]),
    html.Div([
        html.Div([
            html.Label('Select metrics to display:', style={'font-weight': 'bold'}),
            dcc.Checklist(
                id='radar-metrics',
                options=[
                    {'label': 'Exact Match', 'value': 'Exact Match'},
                    {'label': 'f1', 'value': 'f1'},
                    {'label': 'Confidence', 'value': 'Confidence'},
                    {'label': 'Response Time', 'value': 'Response Time'},
                ],
                value=['Exact Match', 'f1', 'Confidence', 'Response Time'],
                style={'border': '1px solid black', 'padding': '5px', 'background-color': '#f9f9f9', 'font-size': '12px'}
            ),
            dcc.Graph(id="metrics-radar-plot")
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label('Select metrics to display:', style={'font-weight': 'bold'}),
            dcc.Checklist(
                id='heatmap-metrics',
                options=[
                    {'label': 'Exact Match', 'value': 'Exact Match'},
                    {'label': 'f1', 'value': 'f1'},
                    {'label': 'Confidence', 'value': 'Confidence'},
                    {'label': 'Response Time', 'value': 'Response Time'}
                ],
                value=['Exact Match', 'f1', 'Confidence', 'Response Time'],
                style={'border': '1px solid black', 'padding': '5px', 'background-color': '#f9f9f9', 'font-size': '12px'}
            ),
            dcc.Graph(id="metrics-heatmap")
        ], style={'width': '48%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            html.Label('Select metrics to display:', style={'font-weight': 'bold'}),
            dcc.Checklist(
                id='pair-metrics',
                options=[
                    {'label': 'Parameter 1', 'value': 'Parameter 1'},
                    {'label': 'Parameter 2', 'value': 'Parameter 2'},
                    {'label': 'Exact Match', 'value': 'Exact Match'},
                    {'label': 'f1', 'value': 'f1'}
                ],
                value=['Parameter 1', 'Parameter 2', 'Exact Match', 'f1'],
                style={'border': '1px solid black', 'padding': '5px', 'background-color': '#f9f9f9', 'font-size': '12px'}
            ),
            dcc.Graph(id="metrics-pair-plot")
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id="metrics-cdf-plot-length")
        ], style={'width': '48%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            dcc.Graph(id="metrics-cdf-plot-confidence")
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id="metrics-cdf-plot-response-time")
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justify-content': 'space-between'})
])



if 'umap_x' not in globals():
    umap_x = np.array([])

if 'umap_y' not in globals():
    umap_y = np.array([])

# Store all outputs and embeddings globally to avoid recomputation
all_outputs = []
all_embeddings = None
all_texts = []

@app.callback(
    [Output("output-container", "children"),
     Output("scatter-plot", "figure"),
     Output("metrics-radar-plot", "figure"),
     Output("metrics-heatmap", "figure"),
     Output("metrics-pair-plot", "figure"),
     Output("metrics-cdf-plot-length", "figure"),
     Output("metrics-cdf-plot-confidence", "figure"),
     Output("metrics-cdf-plot-response-time", "figure"),
     Output("legend-container", "rowData")],
    [Input("submit-button", "n_clicks"),
     Input("radar-metrics", "value"),
     Input("heatmap-metrics", "value"),
     Input("pair-metrics", "value"),
     Input("legend-container", "selectedRows")],
    [State("input-prompt", "value"),
     State("legend-container", "rowData")]
)
def update_output(n_clicks, radar_metrics, heatmap_metrics, pair_metrics, selected_rows, value, current_row_data):
    global all_outputs
    global all_embeddings
    global all_texts

    ctx = dash.callback_context
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

    outputs = {}
    output = ""

    scatter_fig = go.Figure()
    
    if n_clicks > 0 and value and triggered_input == "submit-button":
        outputs = process_prompt(value)
        avg_metrics = calculate_metrics(outputs)
        
        current_metrics = {
            'Metric': ['Exact Match', 'f1', 'Confidence', 'Response Time'],
            'Value': [avg_metrics['exact_match'], avg_metrics['f1'], avg_metrics['confidence'], avg_metrics['response_time']],
            'Prompt': f'Prompt {n_clicks}'  # Add an identifier for the prompt
        }
        previous_metrics.append(current_metrics)
        prompt_contents[f'Prompt {n_clicks}'] = value

        # Append current outputs to the global variable
        all_outputs.append({
            'prompt': f"Prompt {n_clicks}",
            'outputs': outputs,
            'ground_truth': {q: ground_truth_data[q]['output'] for q in outputs.keys()}
        })

        # Update all_texts and all_embeddings
        for entry in all_outputs:
            all_texts.extend(list(entry['outputs'].values()))
            all_texts.extend(entry['ground_truth'].values())

        new_embeddings = np.vstack([calculate_sentence_embedding(text) for text in all_texts])
        if all_embeddings is None:
            all_embeddings = new_embeddings
        else:
            all_embeddings = np.vstack((all_embeddings, new_embeddings))

        umap_x, umap_y = calculate_umap(all_embeddings)

    # Prepare scatter plot
    index = 0
    scatter_fig = go.Figure()

    for i, entry in enumerate(all_outputs):
        prompt_label = entry['prompt']
        outputs = entry['outputs']
        ground_truth = entry['ground_truth']

        response_x, response_y = umap_x[index:index+len(outputs)], umap_y[index:index+len(outputs)]
        index += len(outputs)
        if i == 0:
            gt_x, gt_y = umap_x[index:index+len(ground_truth)], umap_y[index:index+len(ground_truth)]
        index += len(ground_truth)
        color = color_palette[i % len(color_palette)]

        for idx, (question, data) in enumerate(outputs.items()):
            question_label = f"{prompt_label} - Q{idx+1}"
            scatter_fig.add_trace(go.Scatter(
                x=[response_x[idx]], y=[response_y[idx]], mode='markers+text',
                name=f"{question_label} {question}", text=[question_label], textposition='top center',
                hovertext=[data['answer']], hoverinfo='text',
                marker=dict(color=color)
            ))
            scatter_fig.add_trace(go.Scatter(
                x=[gt_x[idx]], y=[gt_y[idx]], mode='markers+text',
                name=f"GT - Q{idx+1}", text=[f"GT - Q{idx+1}"], textposition='top center',
                hovertext=[ground_truth[question]], hoverinfo='text',
                marker=dict(color='green', symbol='diamond')
            ))
            scatter_fig.add_trace(go.Scatter(
                x=[response_x[idx], gt_x[idx]], y=[response_y[idx], gt_y[idx]], mode='lines',
                line=dict(color='gray', dash='dash')
            ))

    scatter_fig.update_layout(title='Sentence Transformer Embeddings UMAP Scatter Plot')

    # If the legend was updated, use the selected prompts
    if selected_rows:
        selected_prompt_keys = [row['prompt'] for row in selected_rows]
    else:
        selected_prompt_keys = list(prompt_contents.keys())

    # Filter metrics based on selected prompts
    filtered_metrics = [metrics for metrics in previous_metrics if metrics['Prompt'] in selected_prompt_keys]

    # Radar plot
    radar_fig = go.Figure()
    if radar_metrics:
        for metrics in filtered_metrics:
            filtered_values = [metrics['Value'][metrics['Metric'].index(metric)] for metric in radar_metrics]
            filtered_metrics_labels = [metric for metric in radar_metrics]
            radar_fig.add_trace(go.Scatterpolar(
                r=filtered_values,
                theta=filtered_metrics_labels,
                fill='toself',
                name=f"{metrics['Prompt']}"
            ))

    radar_fig.update_layout(title='Radar Plot of Metrics')

    # Heatmap
    heatmap_data = {metrics['Prompt']: [metrics['Value'][metrics['Metric'].index(metric)] for metric in heatmap_metrics] for metrics in filtered_metrics}
    heatmap_df = pd.DataFrame(heatmap_data, index=heatmap_metrics)
    heatmap_fig = px.imshow(heatmap_df, text_auto=True, title='Heatmap of Metrics')

    # Pair plot
    pair_plot_data = {
        'Parameter 1': np.random.rand(10),
        'Parameter 2': np.random.rand(10),
        'Exact Match': np.random.rand(10),
        'f1': np.random.rand(10)
    }
    pair_plot_df = pd.DataFrame(pair_plot_data)
    if pair_metrics:
        pair_plot_fig = px.scatter_matrix(pair_plot_df, dimensions=pair_metrics, title='Pair Plot of Parameters and Metrics')
        pair_plot_fig.update_layout(height=800)  # Increase height for more vertical spacing

    # CDF plots
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

    # Legend items
    legend_items = [
        {"prompt": prompt, "content": content, "show": True}
        for prompt, content in prompt_contents.items()
    ]

    return (f"Generated Output: {outputs}", scatter_fig, radar_fig, heatmap_fig, pair_plot_fig, 
            cdf_fig_length, cdf_fig_confidence, cdf_fig_response_time, legend_items)









if __name__ == '__main__':
    app.run_server(debug=False)
