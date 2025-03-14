import sys
import logging
from model import VyomChatbot
from flask import Flask, request, jsonify
from utils.jsonloader import load_json

app = Flask(__name__)
chatbot = VyomChatbot()

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query')
    response = chatbot.ask_question(query)
    return jsonify({"response": response})

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
        dataset = load_json("./dataset/new_dataset.json")

        # Initialize Chatbot
        logger.info("Training Chatbot...")
        chatbot.store_data(dataset)
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
    # app.run(host='0.0.0.0', port=5000)