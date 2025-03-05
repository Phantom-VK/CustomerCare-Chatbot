import sys
import logging
from model import VyomChatbot
from utils.jsonloader import load_json


def main():
    """
    Main function to run the Vyom Chatbot.
    """
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        logger = logging.getLogger(__name__)

        # Load dataset
        questions = load_json("./dataset/questions.json")
        answers = load_json("./dataset/answers.json")

        # Initialize Chatbot
        logger.info("Training Chatbot...")
        chatbot = VyomChatbot()
        chatbot.store_data(questions, answers)
        logger.info("✅ Chatbot is Ready!")

        # Interactive Chat Loop
        while True:
            try:
                query = input("You: ").strip()

                if query.lower() in ['exit', 'quit', 'bye']:
                    logger.info("Chatbot session ended.")
                    print("Goodbye!")
                    break

                response = chatbot.ask_question(query)
                print("Chatbot:", response)

            except KeyboardInterrupt:
                logger.info("Chatbot session interrupted.")
                print("\nSession interrupted. Exiting...")
                break

    except Exception as e:
        logging.error(f"Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()