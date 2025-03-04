from main import model, collection

query = "How can I reset my password?"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print("Answer:", results["documents"][0][0])

