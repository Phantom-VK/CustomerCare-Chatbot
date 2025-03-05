import os
import chromadb
from sentence_transformers import SentenceTransformer

from utils.jsonloader import load_json

# Set custom directory for model downloads
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"

# Now download the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# text = "What is UBI Vyom app?"
# text2 = "UBI Vyom app in android?"
# embedding = model.encode(text)
# embedding2 = model.encode(text2)
# print(embedding.shape)  # (384,)
# print(embedding2.shape)
#
# #Start CHROMA DB
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("faq")


questions = load_json("./dataset/questions.json")
answers = load_json("./dataset/answers.json")

# print(len(questions), len(answers))

for i, question in enumerate(questions):
    embedding = model.encode(question).tolist()
    collection.add(documents=[answers[i]], embeddings=[embedding], ids=[str(i)])
