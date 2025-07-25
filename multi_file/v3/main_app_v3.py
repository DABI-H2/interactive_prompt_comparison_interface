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
import dash_ag_grid as dag
import evaluate
import time
from umap import UMAP
import torch
import random
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

set_seed(seed_value=seed_value)

color_palette = colors.qualitative.Plotly  # You can choose other palettes like D3, G10, etc.

# Load the Sentence Transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_model.eval()

def calculate_sentence_embedding(input_text):
    set_seed(seed_value=seed_value)
    return sentence_model.encode([input_text])

def calculate_clip(input_text, max_tokens=77):
    set_seed(seed_value=seed_value)
    return calculate_sentence_embedding(input_text)

def calculate_umap(embeddings, n_neighbors):
    umap_embeddings = UMAP(metric='cosine', n_components=2, n_neighbors=n_neighbors).fit_transform(embeddings)
    umap_x, umap_y = umap_embeddings[:, 0], umap_embeddings[:, 1]
    return umap_x, umap_y

# Load SQuAD dataset
squad_v2_path = r"C:\Users\desim\Documents\UvA\MMA\SQUAD\dev-v2.0.json"
with open(squad_v2_path) as f:
    squad_data = json.load(f)

cnn_dailymail_data = load_dataset('cnn_dailymail', '3.0.0')
print(cnn_dailymail_data)

# Load evaluation metrics
squad_metric = evaluate.load('squad_v2')

# Prepare the ground truth data
ground_truth_data = {}
offset = 15
n_articles = 2
current_article = 0

for article in cnn_dailymail_data['test']:
    current_article += 1
    if current_article <= offset:
        continue
    if current_article > n_articles + offset:
        break
    context = article['article']
    summary = article['highlights']
    ground_truth_data[f"Article {article['id']}"] = {
        "output": summary,
        "context": context,
    }

previous_metrics = []
prompt_contents = {}

# Load pre-trained GPT model and tokenizer
model_name = 'gpt2'
gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt_model = GPT2LMHeadModel.from_pretrained(model_name).to('cuda')

def process_prompt(prompt):
    outputs = {}
    max_new_tokens = 200
    max_length = 1024 - max_new_tokens  # typically 1024 for this model
    delimiter = " [ANSWER]: "  # Use a special token to separate the prompt and the answer
    temperature = 0.8  # Adjust the temperature to control randomness
    top_k = 50  # Limit the next token selection to the top k most probable tokens
    top_p = 0.9  # Use nucleus sampling

    for question, data in ground_truth_data.items():
        context = data["context"]
        prompt_addition = f"Prompt: {prompt}"
        full_prompt = f"Context: {context}"
        start_time = time.time()
        
        # Tokenize and generate response
        input_ids = gpt_tokenizer.encode(full_prompt, return_tensors='pt').to('cuda')
        delimiter_ids = gpt_tokenizer.encode(delimiter, return_tensors='pt').to('cuda')
        prompt_addition_ids = gpt_tokenizer.encode(prompt_addition, return_tensors='pt').to('cuda')
        input_length = input_ids.shape[1]
        total_length = input_length + len(delimiter) + len(prompt_addition)
        if total_length > max_length:
            print(f"Warning: Input length ({total_length}) exceeds model maximum ({max_length}). Truncating input.")
            input_ids = input_ids[:, :max_length - len(delimiter) - len(prompt_addition)]
        input_ids = torch.cat((input_ids, prompt_addition_ids, delimiter_ids), dim=1)
        input_length = input_ids.shape[1]

        # Ensure the input tensor shape is correct before generating
        assert input_length <= max_length, f"Input tensor shape {input_ids.shape} exceeds the maximum length {max_length}"

        output = gpt_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=gpt_tokenizer.eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )
        
        # Remove input tokens from output
        generated_tokens = output[0][input_length:]  # Slice the output to exclude the input tokens
        generated_text = gpt_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        new_tokens = generated_text.strip()
        
        # Debug print statements
        print(f"Input: {full_prompt}")
        print(f"Generated output tokens (excluding input): {generated_tokens}")
        print(f"Generated text: {generated_text}")

        # Store the generated text as answer
        outputs[question] = {
            'answer': new_tokens,
            'response_time': response_time,
            'confidence': 0  # GPT-2 doesn't provide a confidence score directly
        }
    return outputs

def calculate_metrics(outputs):
    total_rouge_1 = 0
    total_rouge_2 = 0
    total_rouge_l = 0
    total_bleu = 0
    total_confidence = 0
    total_response_time = 0
    num_questions = len(outputs)

    rouge = Rouge()
    for question, output in outputs.items():
        ground_truth = ground_truth_data[question]['output']
        rouge_scores = rouge.get_scores(output['answer'], ground_truth)[0]
        rouge_1 = rouge_scores['rouge-1']['f']
        rouge_2 = rouge_scores['rouge-2']['f']
        rouge_l = rouge_scores['rouge-l']['f']
        bleu_score = sacrebleu.sentence_bleu(output['answer'], [ground_truth]).score / 100
        confidence = output['confidence']
        response_time = output['response_time']

        total_rouge_1 += rouge_1
        total_rouge_2 += rouge_2
        total_rouge_l += rouge_l
        total_bleu += bleu_score
        total_confidence += confidence
        total_response_time += response_time

    avg_rouge_1 = total_rouge_1 / num_questions
    avg_rouge_2 = total_rouge_2 / num_questions
    avg_rouge_l = total_rouge_l / num_questions
    avg_bleu = total_bleu / num_questions
    avg_confidence = total_confidence / num_questions
    avg_response_time = total_response_time / num_questions

    return {
        "ROUGE-1": avg_rouge_1,
        "ROUGE-2": avg_rouge_2,
        "ROUGE-L": avg_rouge_l,
        "BLEU": avg_bleu,
        "Confidence": avg_confidence,
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
    dcc.Tabs([
        dcc.Tab(label='Scatter Plot', children=[
            dcc.Graph(id="scatter-plot")
        ]),
        dcc.Tab(label='Radar Plot', children=[
            html.Div([
                html.Label('Select metrics to display:', style={'font-weight': 'bold'}),
                dcc.Checklist(
                    id='radar-metrics',
                    options=[
                        {'label': 'ROUGE-1', 'value': 'ROUGE-1'},
                        {'label': 'ROUGE-2', 'value': 'ROUGE-2'},
                        {'label': 'ROUGE-L', 'value': 'ROUGE-L'},
                        {'label': 'BLEU', 'value': 'BLEU'},
                        {'label': 'Confidence', 'value': 'Confidence'},
                    ],
                    value=['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'Confidence'],
                    style={'border': '1px solid black', 'padding': '5px', 'background-color': '#f9f9f9', 'font-size': '12px'}
                ),
                dcc.Graph(id="metrics-radar-plot")
            ]),
        ]),
        dcc.Tab(label='Heatmap', children=[
            html.Div([
                html.Label('Select metrics to display:', style={'font-weight': 'bold'}),
                dcc.Checklist(
                    id='heatmap-metrics',
                    options=[
                        {'label': 'ROUGE-1', 'value': 'ROUGE-1'},
                        {'label': 'ROUGE-2', 'value': 'ROUGE-2'},
                        {'label': 'ROUGE-L', 'value': 'ROUGE-L'},
                        {'label': 'BLEU', 'value': 'BLEU'},
                        {'label': 'Confidence', 'value': 'Confidence'}
                    ],
                    value=['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'Confidence'],
                    style={'border': '1px solid black', 'padding': '5px', 'background-color': '#f9f9f9', 'font-size': '12px'}
                ),
                dcc.Graph(id="metrics-heatmap")
            ]),
        ]),
        dcc.Tab(label='Pair Plot', children=[
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
            ])
        ]),
        dcc.Tab(label='CDF Plots', children=[
            html.Div([
                dcc.Graph(id="metrics-cdf-plot-length")
            ]),
            html.Div([
                dcc.Graph(id="metrics-cdf-plot-confidence")
            ]),
            html.Div([
                dcc.Graph(id="metrics-cdf-plot-response-time")
            ])
        ]),
        dcc.Tab(label='Legend and Article Navigation', children=[
            html.A(
                "Open Legend and Article Navigation",
                href="/legend",
                target="_blank",
                className="button"
            )
        ])
    ])
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
     Output("legend-container", "rowData"),
     Output('article-content', 'children'),
     Output('article-responses', 'children'),
     Output('prompt-dropdown', 'options'),
     Output('prompt-dropdown', 'value')],
    [Input("submit-button", "n_clicks"),
     Input("radar-metrics", "value"),
     Input("heatmap-metrics", "value"),
     Input("pair-metrics", "value"),
     Input("legend-container", "selectedRows"),
     Input('article-id-dropdown', 'value')],
    [State("input-prompt", "value"),
     State("legend-container", "rowData"),
     State("prompt-dropdown", "value")]
)
def update_output(n_clicks, radar_metrics, heatmap_metrics, pair_metrics, selected_rows, article_id, value, current_row_data, selected_prompt):
    global all_outputs
    global all_embeddings
    global all_texts
    global umap_x
    global umap_y

    ctx = dash.callback_context
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

    outputs = {}
    output = ""

    scatter_fig = go.Figure()
    
    if n_clicks > 0 and value and triggered_input == "submit-button":
        outputs = process_prompt(value)
        avg_metrics = calculate_metrics(outputs)
        
        current_metrics = {
            'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'Confidence'],
            'Value': [avg_metrics['ROUGE-1'], avg_metrics['ROUGE-2'], avg_metrics['ROUGE-L'], avg_metrics['BLEU'], avg_metrics['Confidence']],
            'Prompt': f'Prompt {n_clicks}'  # Add an identifier for the prompt
        }
        previous_metrics.append(current_metrics)
        prompt_contents[f'Prompt {n_clicks}'] = value

        # Append current outputs to the global variable
        all_outputs.append({
            'prompt': f"Prompt {n_clicks}",
            'outputs': {q: outputs[q]['answer'] for q in outputs.keys()},
            'ground_truth': {q: ground_truth_data[q]['output'] for q in outputs.keys()}
        })

        # Update all_texts and all_embeddings
        for entry in all_outputs:
            all_texts.extend(list(entry['outputs'].values()))
            all_texts.extend(entry['ground_truth'].values())
        print("ALL TEXTS:", all_texts)
        new_embeddings = np.vstack([calculate_clip(text) for text in all_texts])
        print("NEW EMBEDDINGS:", new_embeddings)
        if all_embeddings is None:
            all_embeddings = new_embeddings
        else:
            all_embeddings = np.vstack((all_embeddings, new_embeddings))

        # Ensure that all_embeddings is not empty before calculating UMAP
        if all_embeddings.size > 0:
            umap_x, umap_y = calculate_umap(all_embeddings, n_neighbors=max(len(all_outputs), 2))
        else:
            umap_x, umap_y = np.array([]), np.array([])

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
                hovertext=[data], hoverinfo='text',
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
                line=dict(color='gray', dash='dash'),
                showlegend=False  # Add this line
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
            filtered_metrics_labels = [metric for metric in radar_metrics if metric != 'Response Time']
            radar_fig.add_trace(go.Scatterpolar(
                r=filtered_values,
                theta=filtered_metrics_labels,
                fill='toself',
                name=f"{metrics['Prompt']}"
            ))

    radar_fig.update_layout(title='Radar Plot of Metrics')

    # Heatmap
    heatmap_data = {metrics['Prompt']: [metrics['Value'][metrics['Metric'].index(metric)] for metric in heatmap_metrics if metric != 'Response Time'] for metrics in filtered_metrics}
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

    # Article content and responses
    if article_id:
        article_content = ground_truth_data[article_id]['context']
        article_responses = ""
        for entry in all_outputs:
            for q, response in entry['outputs'].items():
                if q in ground_truth_data and ground_truth_data[q]['context'] == article_content:
                    article_responses += f"{entry['prompt']}: {response}\n"
        prompt_options = [{'label': prompt, 'value': prompt} for prompt in prompt_contents.keys()]
        prompt_value = selected_prompt if selected_prompt in prompt_contents else None
    else:
        article_content = ""
        article_responses = ""
        prompt_options = []
        prompt_value = None

    return (f"Generated Output: {outputs}", scatter_fig, radar_fig, heatmap_fig, pair_plot_fig, 
            cdf_fig_length, cdf_fig_confidence, cdf_fig_response_time, legend_items, article_content, article_responses,
            prompt_options, prompt_value)


if __name__ == '__main__':
    app.run_server(debug=False)
