# Chatbot using Cosine Similarity

## Introduction
This project is a simple chatbot built using cosine similarity to match user input with the most similar pre-defined conversation.

## Features
- Uses TF-IDF Vectorization to convert text data into numerical format.
- Computes cosine similarity between user input and pre-defined conversations to find the best match.
- Provides a response corresponding to the most similar pre-defined conversation.

## Dependencies
- numpy
- sklearn

## How to Use
1. Clone this repository.
2. Install the dependencies using pip:
3. Run the script:
4. Interact with the chatbot in the console by typing your input after "You: ". To exit the chatbot, type 'exit'.

## Code Overview
The code includes a dataset of pre-defined conversations, each with an input and a corresponding response. The `TfidfVectorizer` from `sklearn.feature_extraction.text` is used to convert the input text data into numerical format. The `cosine_similarity` function from `sklearn.metrics.pairwise` is used to compute the similarity between the user's input and the pre-defined conversations. The response corresponding to the most similar pre-defined conversation is returned as the chatbot's response.

## Future Work
- Expand the dataset of pre-defined conversations for a wider range of responses.
- Implement a more sophisticated matching algorithm for improved conversation quality.
