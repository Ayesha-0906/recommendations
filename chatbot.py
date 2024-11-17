from flask import Flask, request, jsonify
import pandas as pd
import pickle
from fuzzywuzzy import process

app = Flask(__name__)

# Load model and data
kmeans = pickle.load(open("kmeans.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
data = pd.read_csv("processed_data.csv")

def recommend(place_name, num_recommendations=5):
    # Normalize dataset place names
    data['Normalized_Place'] = data['Place'].str.strip().str.lower()
    
    # Normalize the input place name
    place_name = place_name.strip().lower()
    
    # Use fuzzy matching to find the closest place in the dataset
    best_match = process.extractOne(place_name, data['Normalized_Place'])
    
    # If fuzzy match score is above threshold (80), use the match, otherwise return an error
    if best_match and best_match[1] > 80:
        matched_place = best_match[0]  # Best matching place from dataset
        cluster = data[data['Normalized_Place'] == matched_place]['Cluster'].values[0]  # Get cluster of matched place
    else:
        return ["Place not found in the dataset."]
    
    # Recommend places in the same cluster
    cluster_places = data[data['Cluster'] == cluster]  # Get places in the same cluster
    recommendations = cluster_places[cluster_places['Normalized_Place'] != matched_place].head(num_recommendations)
    
    # Return the recommended places
    return recommendations['Place'].tolist()

@app.route('/recommend', methods=['POST'])
def chatbot():
    user_data = request.json
    place = user_data.get('place', '')  # Get place from JSON request
    recommendations = recommend(place)  # Get recommendations
    return jsonify({"recommendations": recommendations})  # Return recommendations as JSON response

if __name__ == "__main__":
    app.run(debug=True)
