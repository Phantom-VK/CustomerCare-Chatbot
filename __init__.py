from model import model, collection


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question: ")
        query_embedding = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )
        print("Answer:", results["documents"][0][0])
