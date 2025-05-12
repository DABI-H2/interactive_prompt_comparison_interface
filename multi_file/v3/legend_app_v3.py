import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_ag_grid as dag
import json
from datasets import load_dataset

# Load SQuAD dataset
squad_v2_path = r"C:\Users\desim\Documents\UvA\MMA\SQUAD\dev-v2.0.json"
with open(squad_v2_path) as f:
    squad_data = json.load(f)

cnn_dailymail_data = load_dataset('cnn_dailymail', '3.0.0')
print(cnn_dailymail_data)

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

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Legend', children=[
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
        ]),
        dcc.Tab(label='Article Navigation', children=[
            html.Div([
                html.Label('Article ID:', style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='article-id-dropdown',
                    options=[
                        {'label': key, 'value': key} for key in ground_truth_data.keys()
                    ],
                    style={'width': '80%'}
                ),
                html.Br(),
                html.Label('Prompt:', style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='prompt-dropdown',
                    style={'width': '80%'}
                ),
                html.Br(),
                html.Label('Article Content:', style={'font-weight': 'bold'}),
                html.Div(id='article-content', style={'whiteSpace': 'pre-wrap', 'border': '1px solid #ccc', 'padding': '10px', 'width': '80%', 'margin': 'auto'}),
                html.Br(),
                html.Label('Responses:', style={'font-weight': 'bold'}),
                html.Div(id='article-responses', style={'whiteSpace': 'pre-wrap', 'border': '1px solid #ccc', 'padding': '10px', 'width': '80%', 'margin': 'auto'})
            ])
        ])
    ])
])

@app.callback(
    [Output('article-content', 'children'),
     Output('article-responses', 'children'),
     Output('prompt-dropdown', 'options'),
     Output('prompt-dropdown', 'value')],
    [Input('article-id-dropdown', 'value')],
    [State('prompt-dropdown', 'value')]
)
def update_article(article_id, selected_prompt):
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

    return article_content, article_responses, prompt_options, prompt_value

if __name__ == '__main__':
    app.run_server(debug=False, port=8051)
