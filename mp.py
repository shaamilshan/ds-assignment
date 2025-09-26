import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (replace with your dataset)
movies = pd.read_csv('tmdb_5000_movies.csv')



# Extract main genre from the 'genres' column (JSON-like string)
import ast
def extract_main_genre(genres_str):
	try:
		genres_list = ast.literal_eval(genres_str)
		if isinstance(genres_list, list) and len(genres_list) > 0:
			return genres_list[0]['name']
		else:
			return 'Unknown'
	except Exception:
		return 'Unknown'

movies['main_genre'] = movies['genres'].apply(extract_main_genre)
le = LabelEncoder()
movies['genre_encoded'] = le.fit_transform(movies['main_genre'])


# Select features and target (update as needed)
features = ['budget', 'genre_encoded']  
X = movies[features]
y = movies['vote_average'] > 7  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict probability of success (chance)
success_proba = model.predict_proba(X_test)[:, 1] 

# Get the corresponding movie titles for X_test
X_test_with_titles = X_test.copy()
X_test_with_titles['title'] = movies.loc[X_test_with_titles.index, 'title'].values
X_test_with_titles['success_chance'] = success_proba

print("\nSample of predicted success chances:")
print(X_test_with_titles[['title', 'success_chance']].head(10))

# Show top 10 movies with highest predicted success chance
print("\nTop 10 movies with highest predicted success chance:")
print(X_test_with_titles[['title', 'success_chance']].sort_values('success_chance', ascending=False).head(10))

