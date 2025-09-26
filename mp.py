import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
movies = pd.read_csv('tmdb_5000_movies.csv')

# Use simple numeric features
X = movies[['budget', 'popularity']].fillna(0)
y = movies['vote_average'] > 7  # Success = rating > 7

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict probabilities
success_proba = model.predict_proba(X_test)[:, 1]  

# Add titles back with success %
results = pd.DataFrame({
    'title': movies.loc[X_test.index, 'title'],
    'success_chance': success_proba * 100  
})

# Show sample
print(results.head(10))

# Show top 10 most likely to succeed
print("\nTop 10 movies with highest success %:")
print(results.sort_values('success_chance', ascending=False).head(10))
