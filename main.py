import os
import chromadb
from sentence_transformers import SentenceTransformer

# Set custom directory for model downloads
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"

# Now download the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# text = "What is UBI Vyom app?"
# embedding = model.encode(text)
# # print(embedding.shape)  # (384,)

#Start CHROMA DB
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("faq")

questions = [
    "What is UBI Vyom?",
    "How to reset password?",
    "How to contact customer care?"
]
answers = [
    "UBI Vyom is the official app of Union Bank of India.",
    "Go to settings and select reset password.",
    "Call 1800-1234-567 or email customercare@ubi.com."
]

for q, a in zip(questions, answers):
    embedding = model.encode(q).tolist()
    collection.add(documents=[a], embeddings=[embedding], ids=[q])