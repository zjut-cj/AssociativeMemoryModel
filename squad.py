import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Example data from SQuAD dataset
context = "Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity."
question = "What is Albert Einstein known for?"
answer = "the theory of relativity"

# Tokenize context and question
input_text = context + " [SEP] " + question
input_ids = tokenizer.encode(input_text, add_special_tokens=True)

# Convert to PyTorch tensor
input_ids = torch.tensor(input_ids)

# Get embeddings from BERT model
with torch.no_grad():
    outputs = model(input_ids.unsqueeze(0))  # Batch size 1
    embeddings = outputs[0]

# embeddings contains the contextual embeddings of the input text
# You can use these embeddings for various downstream tasks

# If you want to extract embeddings for specific words or tokens, you can do so from embeddings
# For example, if you want the embeddings for the word "Einstein"
einstein_embeddings = embeddings[0, input_ids.index(tokenizer.convert_tokens_to_ids("Einstein"))]

print(einstein_embeddings)
