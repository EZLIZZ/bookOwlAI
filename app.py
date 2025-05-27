from flask import Flask, jsonify, request
from flask_cors import CORS  # ✅ Import CORS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
# Initialize Flask
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all origins (you can restrict it later)


# MongoDB Connection
MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_NAME = os.getenv("MONGODB_NAME")

client = MongoClient(MONGODB_URL)
db = client[MONGODB_NAME]

# Fetch books
collection = db["books"]
books = list(collection.find({}, {"_id": 1, "ISBN": 1, "bookName": 1, "coverImage": 1, "ratingsCount": 1, "averageRating": 1}))
books_df = pd.DataFrame(books)

# Fetch ratings
rev_collection = db["reviews"]
ratings = list(rev_collection.find({}, {"_id": 0, "book": 1, "user": 1, "rating": 1}))
ratings_df = pd.DataFrame(ratings)

# Merge books and ratings
if not ratings_df.empty and not books_df.empty:
    books_df['_id'] = books_df['_id'].astype(str)
    ratings_df['book'] = ratings_df['book'].astype(str)
    merged_df = ratings_df.merge(books_df, left_on="book", right_on="_id", how="inner")

    # Pivot table for collaborative filtering
    pt = merged_df.pivot_table(index='_id', columns='user', values='rating', fill_value=0)

    # Compute similarity
    similarity_scores = cosine_similarity(pt)

else:
    merged_df = pd.DataFrame()
    similarity_scores = None

@app.route('/')
def home():
    return "Welcome to the AI Backend!"

# Function to recommend books
# Function to recommend books
def recommend(book_id):
    book_id = str(book_id).strip()  # Ensure it's a clean string
    
    if similarity_scores is None or pt.empty:
        return {"error": "No data available for recommendations."}

    pt_index = pt.index.astype(str)

    if book_id not in pt_index:
        return {"error": f"Book ID {book_id} not found in the index."}
    
    # Get the index of the book in the similarity matrix
    index = np.where(pt_index == book_id)[0][0]
    similarity_scores_for_book = similarity_scores[index]

    # Get all similar books sorted by similarity score
    similar_items = sorted(
        list(enumerate(similarity_scores_for_book)), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Remove the book itself from the recommendations
    similar_items = [item for item in similar_items if pt.index[item[0]] != book_id]
    
    # Get top 5 similar books
    similar_items = similar_items[:5]

    # Get recommended books data
    recommended_books = []
    for i, score in similar_items:
        recommended_id = str(pt.index[i])
        book_data = books_df.loc[books_df['_id'] == recommended_id]
        
        if not book_data.empty:
            recommended_books.append({
                "book_id": recommended_id,
                "bookName": book_data['bookName'].values[0],
                "coverImage": book_data['coverImage'].values[0],
                "similarity_score": float(score)  # optional for debugging
            })

    return recommended_books


# API Route for Recommendations
@app.route("/recommend/<book_id>", methods=["GET"])
def api_recommend(book_id):
    recommendations = recommend(book_id)
    
    if "error" in recommendations:
        return jsonify(recommendations), 404
    
    return jsonify({"recommendations": recommendations})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
