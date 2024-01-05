from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Assuming 'conversations' is your dataset
conversations = [
    {"input": "Hello", "response": "Hi, how can I help you?"},
    {"input": "What's the weather like?", "response": "I'm sorry, I can't provide real-time information."},
    {"input": "Tell me a joke", "response": "Why don't scientists trust atoms? Because they make up everything!"},
    {"input": "Who won the world cup?", "response": "I'm sorry, I can't provide real-time information."},
    {"input": "Tell me a story", "response": "Once upon a time, in a land far, far away..."},
    {"input": "What is AI?", "response": "AI, or Artificial Intelligence, is a field of computer science that involves creating machines that mimic human intelligence."},
    {"input": "Who are you?", "response": "I'm an AI-powered assistant."},
    {"input": "What can you do?", "response": "I can provide information, answer questions, tell jokes, and much more!"},
    {"input": "Goodbye", "response": "Goodbye! Have a great day!"},
    # Add more conversations here...
]

# We'll create our X and y pairs
X = [conv["input"] for conv in conversations]
y = [conv["response"] for conv in conversations]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the data
X_vectorized = vectorizer.fit_transform(X)

def get_response(user_input):
    # Vectorize the user's input
    user_input_vectorized = vectorizer.transform([user_input])

    # Compute cosine similarity between the user's input and the dataset
    similarity_scores = cosine_similarity(user_input_vectorized, X_vectorized)

    print(similarity_scores)

    # Get the index of the most similar input in the dataset
    most_similar_index = np.argmax(similarity_scores)

    print(most_similar_index)
    # Return the corresponding response
    return y[most_similar_index]

# Test the function
while True:
    # Get user input
    user_input = input("You: ")

    # Check for exit condition
    if user_input.lower() == 'exit':
        print("Exiting the chatbot.")
        break

    # Get and print the model's response
    response = get_response(user_input)
    print("Chatbot:", response)


