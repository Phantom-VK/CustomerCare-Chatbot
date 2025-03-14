import os
import logging
import torch

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set custom model directory
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"
os.environ["TRANSFORMERS_CACHE"] = "./models"

class VyomChatbot:
    def __init__(self,
                 embedding_model_name: str = "all-mpnet-base-v2",
                 db_path: str = "./db",
                 collection_name: str = "faq",
                 similarity_threshold: float = 0.2):
        """
        Initialize the Vyom Chatbot with configurable models and settings.

        Args:
            embedding_model_name (str): Name of embedding model
            db_path (str): Path to ChromaDB database
            collection_name (str): Name of ChromaDB collection
            similarity_threshold (float): Minimum similarity score for retrieval
        """
        try:
            # Check if CUDA is available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            # Load Embedding Model
            logger.info("Loading Embedding Model...")
            self.embedding_model = SentenceTransformer(embedding_model_name).to(self.device)


            # Initialize ChromaDB
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(collection_name)

            # Set similarity threshold
            self.similarity_threshold = similarity_threshold

        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            raise

    def store_data(self, data: List[Dict[str, List[str]]]) -> None:
        """
        Store questions and answers in ChromaDB with embeddings.

        Args:
            data (List[Dict[str, List[str]]]): List of dictionaries containing "intent", "questions", and "answer"
        """
        try:
            for item in data:
                intent = item["intent"]
                questions = item["questions"]
                answer = item["answer"]

                # Generate embeddings for each question and store them with the answer
                for i, question in enumerate(questions):
                    embedding = self.embedding_model.encode(question, device=self.device).tolist()
                    self.collection.add(
                        documents=answer,
                        embeddings=[embedding],
                        ids=[f"{intent}_{i}"]
                    )
            logger.info(f"Stored {len(data)} intents with questions and answers.")
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            raise

    def ask_question(self, query: str) -> str:
        """
        Find the most similar answer or generate a response.

        Args:
            query (str): User's input query

        Returns:
            str: Most similar answer or generated response
        """
        try:
            query_embedding = self.embedding_model.encode(query, device=self.device).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1
            )

            # Check similarity score
            similarity_score = results["distances"][0][0]

            if similarity_score > self.similarity_threshold:
                return results["documents"][0][0]
            else:
                # Provide a predefined fallback response
                return "I'm sorry, I couldn't find the answer to your query. Please contact customer support for further assistance."

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I'm sorry, I couldn't understand your query. Could you please rephrase?"