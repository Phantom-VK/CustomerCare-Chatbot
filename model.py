import os
import logging

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, pipeline
from typing import List


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set custom model directory
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"
os.environ["TRANSFORMERS_CACHE"] = "./models"


class VyomChatbot:
    def __init__(self,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 bert_model_name: str = "bert-base-uncased",
                 generator_model_name: str = "gpt2",
                 db_path: str = "./db",
                 collection_name: str = "faq",
                 similarity_threshold: float = 0.6):
        """
        Initialize the Vyom Chatbot with configurable models and settings.

        Args:
            embedding_model_name (str): Name of embedding model
            bert_model_name (str): Name of BERT model
            generator_model_name (str): Name of text generation model
            db_path (str): Path to ChromaDB database
            collection_name (str): Name of ChromaDB collection
            similarity_threshold (float): Minimum similarity score for retrieval
        """
        try:
            # Load Embedding Model
            logger.info("Loading Embedding Model...")
            self.embedding_model = SentenceTransformer(embedding_model_name)

            # Load BERT Model
            logger.info("Loading BERT Model...")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert_model = AutoModel.from_pretrained(bert_model_name)

            # Load GPT-2 Model
            logger.info("Loading GPT-2 Model...")
            self.generator = pipeline("text-generation", model=generator_model_name)

            # Initialize ChromaDB
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(collection_name)

            # Set similarity threshold
            self.similarity_threshold = similarity_threshold

        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            raise

    def store_data(self, questions: List[str], answers: List[str]) -> None:
        """
        Store questions and answers in ChromaDB with embeddings.

        Args:
            questions (List[str]): List of questions
            answers (List[str]): List of corresponding answers
        """
        try:
            for i, question in enumerate(questions):
                embedding = self.embedding_model.encode(question).tolist()
                self.collection.add(
                    documents=[answers[i]],
                    embeddings=[embedding],
                    ids=[str(i)]
                )
            logger.info(f"Stored {len(questions)} questions and answers.")
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
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1
            )

            # Check similarity score
            similarity_score = results["distances"][0][0]

            if similarity_score < self.similarity_threshold:
                return results["documents"][0][0]
            else:
                # # Generate a fallback response
                # generated_response = self.generator(
                #     query,
                #     max_length=50,
                #     num_return_sequences=1,
                #     temperature=0.7,  # Lower = Less Random
                #     truncation=True,
                #     pad_token_id= self.generator.tokenizer.eos_token_id
                # )[0]["generated_text"]
                return ("I'm sorry, I couldn't find the answer to your query. Please generate a support ticket for "
                        "further assistance.")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I'm sorry, I couldn't understand your query. Could you please rephrase?"