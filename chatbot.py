# Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Predefined questions
questions = [
    "What is AI?",
    "Explain artificial intelligence",
    "What is machine learning?",
    "Explain machine learning",
    "What is Python?",
    "Tell me about Python",
    "What is deep learning?",
    "What is NLP?"
]


# Corresponding answers
answers = [
    "AI means Artificial Intelligence.",
    "Artificial Intelligence enables machines to mimic human intelligence.",
    "Machine learning is a subset of AI.",
    "Machine learning allows systems to learn from data.",
    "Python is a programming language.",
    "Python is widely used for AI, ML, and web development.",
    "Deep learning is a type of machine learning.",
    "NLP stands for Natural Language Processing."
]


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert questions into vectors
X = vectorizer.fit_transform(questions)


# Start chatbot loop
while True:

    # Get user input
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() == "exit":
        print("Chatbot ended.")
        break

    # Convert user input into vector
    user_vec = vectorizer.transform([user_input])

    # Calculate similarity
    similarity = cosine_similarity(user_vec, X)

    # Get highest similarity score
    max_similarity = similarity.max()

    # Get matching question index
    index = similarity.argmax()

    # If similarity is low, chatbot doesn't know answer
    if max_similarity < 0.3:
        print("Bot: Sorry, I don't understand that question.")
    else:
        print("Bot:", answers[index])