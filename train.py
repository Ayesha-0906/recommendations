import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
import os

# Set the number of OpenMP threads to 1 to avoid memory leak on Windows
os.environ["OMP_NUM_THREADS"] = "1"

def preprocess_data(data):
    # One-hot encode categorical features (Category, Best_Season)
    encoder = OneHotEncoder()  # Remove sparse=False
    categorical_features = encoder.fit_transform(data[['Category', 'Best_Season']]).toarray()  # Convert to dense array

    # Standardize numerical features (Cost, Rating, User_Likes)
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(data[['Cost', 'Rating', 'User_Likes']])

    # Combine both features
    processed_data = pd.concat([pd.DataFrame(numerical_features, columns=['Cost', 'Rating', 'User_Likes']),
                                pd.DataFrame(categorical_features, columns=encoder.get_feature_names_out(['Category', 'Best_Season']))], axis=1)
    
    return processed_data, scaler, encoder

def train_model():
    # Load dataset
    data = pd.read_csv("dataset.csv")
    
    # Preprocess the data
    processed_data, scaler, encoder = preprocess_data(data)
    
    # Apply K-Means clustering
    num_clusters = 5  # You can change the number of clusters if needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(processed_data)
    
    # Save processed data, scaler, encoder, and model
    data.to_csv("processed_data.csv", index=False)  # Save processed data with cluster labels
    with open("kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
