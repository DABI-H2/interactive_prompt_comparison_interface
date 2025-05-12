from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

# Function to get BERT embeddings for a given sentence
def get_bert_embedding(sentence, model, tokenizer):
    # Tokenize the sentence
    tokens = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    # Compute the embeddings
    with torch.no_grad():
        outputs = model(**tokens)
    # Get the [CLS] token embedding (sentence embedding)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
    return embedding

# Calculate UMAP embeddings
def calculate_umap(embeddings, n_neighbors):
    umap_embeddings = UMAP(metric='cosine', n_components=2, n_neighbors=n_neighbors).fit_transform(embeddings)
    umap_x, umap_y = umap_embeddings[:, 0], umap_embeddings[:, 1]
    return umap_x, umap_y

# Load the BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# Define three sentences with the same meaning but different lengths
sentence1 = "The cat sat on the mat."
sentence2 = "The feline creature positioned itself comfortably on the soft floor covering."
sentence3 = "A cat found a spot on the rug to sit down."

# Get the embeddings for all sentences
embedding1 = get_bert_embedding(sentence1, model, tokenizer)
embedding2 = get_bert_embedding(sentence2, model, tokenizer)
embedding3 = get_bert_embedding(sentence3, model, tokenizer)

# Convert embeddings to numpy arrays
embedding1_np = embedding1.numpy()
embedding2_np = embedding2.numpy()
embedding3_np = embedding3.numpy()

# Calculate UMAP before stacking
umap_x1, umap_y1 = calculate_umap(np.vstack([embedding1_np, embedding2_np, embedding3_np]), n_neighbors=2)

# Stack the embeddings
embeddings_stacked = torch.vstack([embedding1, embedding2, embedding3])
embeddings_stacked_np = embeddings_stacked.numpy()

# Calculate UMAP after stacking
umap_x2, umap_y2 = calculate_umap(embeddings_stacked_np, n_neighbors=2)

# Plot UMAP before stacking
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(umap_x1, umap_y1, color='blue')
plt.title('UMAP before stacking')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
for i, sentence in enumerate([sentence1, sentence2, sentence3]):
    plt.annotate(f'Sentence {i+1}', (umap_x1[i], umap_y1[i]))

# Plot UMAP after stacking
plt.subplot(1, 2, 2)
plt.scatter(umap_x2, umap_y2, color='red')
plt.title('UMAP after stacking')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
for i, sentence in enumerate([sentence1, sentence2, sentence3]):
    plt.annotate(f'Sentence {i+1}', (umap_x2[i], umap_y2[i]))

plt.tight_layout()
plt.show()
