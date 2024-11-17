import pandas as pd
import pickle
from fuzzywuzzy import process

# Load the dataset
data = pd.read_csv('processed_data.csv')  # Ensure the file path is correct

# Load the KMeans model, scaler, and encoder
with open('kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

def recommend(place_name, interest, budget, group_type, num_recommendations=5):
    # Normalize dataset place names
    data['Normalized_Place'] = data['Place'].str.strip().str.lower()  # Normalize dataset names
    place_name = place_name.strip().lower()  # Normalize the input place name
    
    # Use fuzzy matching to find the closest place in the dataset
    best_match = process.extractOne(place_name, data['Normalized_Place'])
    
    print(f"Best match found for '{place_name}': {best_match}")  # Debugging line
    
    if best_match and best_match[1] > 80:  # Threshold for fuzzy matching
        matched_place = best_match[0]
    else:
        return ['Place not found in the dataset.']
    
    # Find the row for the matched place
    matched_row = data[data['Normalized_Place'] == matched_place].iloc[0]
    
    # Get the cluster number of the matched place
    cluster_number = matched_row['Cluster']
    
    # Filter the dataset based on the cluster number
    recommended_places = data[data['Cluster'] == cluster_number]
    
    # Convert 'Cost' column to numeric for budget filtering
    recommended_places['Cost'] = pd.to_numeric(recommended_places['Cost'], errors='coerce')  # Ensures numeric values
    
    print("Filtered by budget and category:")
    print(recommended_places.head(10))  # Debugging line to check filtering output
    
    # Flexible filtering by budget (handle numerical ranges)
    if budget.lower() != 'any':
        if budget.lower() == 'low':
            recommended_places = recommended_places[recommended_places['Cost'] <= 150]
        elif budget.lower() == 'medium':
            recommended_places = recommended_places[(recommended_places['Cost'] > 150) & (recommended_places['Cost'] <= 300)]
        elif budget.lower() == 'high':
            recommended_places = recommended_places[recommended_places['Cost'] > 300]
    
    # Flexible filtering by interest (e.g., Adventure, Culture, etc.)
    if interest.lower() != 'all':
        recommended_places = recommended_places[recommended_places['Category'].str.contains(interest.lower(), case=False, na=False)]
    
    # Exclude the input place from the recommendations
    recommended_places = recommended_places[recommended_places['Normalized_Place'] != place_name]
    
    # Limit the recommendations to the top N
    recommended_places = recommended_places.head(num_recommendations)
    
    # Return the names of the recommended places
    return recommended_places['Place'].tolist()

# Input from the user
place = input("Enter a place you like: ").strip()  # Remove leading/trailing spaces
interest = input("What type of places are you interested in? (e.g., Beach, Adventure, Culture, All): ").strip()
budget = input("What's your budget range? (e.g., Low, Medium, High, Any): ").strip()
group_type = input("Are you traveling with Family, Friends, or Alone? (Family, Friends, Any): ").strip()

# Get and display recommendations
recommendations = recommend(place, interest, budget, group_type)
print("Recommended Places:")
if recommendations:
    for rec in recommendations:
        print(rec)
else:
    print("No matching places found.")
