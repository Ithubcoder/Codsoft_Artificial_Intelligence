from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load the dataset.
data = Dataset.load_builtin('ml-100k')

# Split the dataset into training and test sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use the SVD algorithm
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

# Function to get movie recommendations for a user
def get_recommendations(user_id, num_recommendations=5):
    # Get a list of all movie IDs
    movie_ids = trainset.all_items()
    movie_ids = [trainset.to_raw_iid(movie_id) for movie_id in movie_ids]
    
    # Predict ratings for all movies
    predictions = [algo.predict(user_id, movie_id) for movie_id in movie_ids]
    
    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get the top N recommendations
    top_predictions = predictions[:num_recommendations]
    
    # Print the top recommendations
    for prediction in top_predictions:
        print(f"Movie ID: {prediction.iid}, Predicted Rating: {prediction.est}")

# Example: Get recommendations for user with user_id = 196
get_recommendations(user_id=196, num_recommendations=5)