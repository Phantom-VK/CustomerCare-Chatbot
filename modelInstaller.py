import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, pipeline

# Set custom directory for all models
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"
os.environ["TRANSFORMERS_CACHE"] = "./models"

# Install Sentence Embedding Model
print("Downloading Sentence Embedding Model...")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Sentence Embedding Model Installed")

# Install BERT Model
print("Downloading BERT Model...")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
print("✅ BERT Model Installed")

# Install GPT-2 Model
print("Downloading GPT-2 Model...")
generator = pipeline("text-generation", model="gpt2")
print("✅ GPT-2 Model Installed")

print("✅ All Models Installed Successfully in ./models")
