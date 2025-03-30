import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from flask import Flask, request, jsonify

# Load and Process Movie Dataset
movies = pd.read_csv("tmdb_5000_movies.csv")

# Keep only relevant columns and drop NaN values
movies = movies[['title', 'overview']].dropna()

# Convert movie descriptions into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['overview'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on input text
def recommend_movies(text, num_recommendations=5):
    text_vector = vectorizer.transform([text])
    similarity_scores = cosine_similarity(text_vector, tfidf_matrix)
    movie_indices = similarity_scores.argsort()[0][-num_recommendations:][::-1]
    return movies.iloc[movie_indices][['title', 'overview']]

# Create and Train ChatBot
bot = ChatBot("MovieBot")
trainer = ChatterBotCorpusTrainer(bot)
trainer.train("chatterbot.corpus.english")

# Function to handle chatbot responses
def chatbot_response(user_input):
    if "recommend" in user_input.lower():
        recommendations = recommend_movies(user_input)
        response = "Here are some movie recommendations:\n" + "\n".join(recommendations['title'].tolist())
    else:
        response = str(bot.get_response(user_input))
    return response

# Create Flask App
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    bot_reply = chatbot_response(user_message)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
