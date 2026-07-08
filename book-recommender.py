import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os

# 0. Safety Check
# This ensures you have placed the files in the correct folder before running
if not os.path.exists('data/BX-Books.csv'):
    print("Error: Could not find the dataset.")
    print("Please make sure you downloaded the data and placed the three CSV files inside a folder named 'data'.")
    exit()

# 1. Load the Data
# 1. Load the Data
print("Loading datasets from 'data/' directory... (This might take a minute)")

# We are just telling Python to look for commas (,) instead of semicolons (;)
books = pd.read_csv('data/BX-Books.csv', sep=',', on_bad_lines='skip', encoding='latin-1', low_memory=False)
users = pd.read_csv('data/BX-Users.csv', sep=',', on_bad_lines='skip', encoding='latin-1', low_memory=False)
ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=',', on_bad_lines='skip', encoding='latin-1', low_memory=False)

# --- DEBUGGING PRINT ---
print("\n--- DEBUG INFO ---")
print("Books columns found:", books.columns.tolist())
print("------------------\n")

# 2. Data Preprocessing & Feature Engineering
print("Cleaning and processing data...")

# Strip any hidden whitespace from column names just in case
books.columns = books.columns.str.strip()
users.columns = users.columns.str.strip()
ratings.columns = ratings.columns.str.strip()

# Rename the columns safely
books.rename(columns={'Book-Title': 'title', 'Book-Author': 'author', 'Year-Of-Publication': 'year', 'Publisher': 'publisher'}, inplace=True)
users.rename(columns={'User-ID': 'user_id', 'Location': 'location', 'Age': 'age'}, inplace=True)
ratings.rename(columns={'User-ID': 'user_id', 'Book-Rating': 'rating'}, inplace=True)

# Keep only necessary columns for books
books = books[['ISBN', 'title', 'author', 'year', 'publisher']]

# 3. Filtering Data (Collaborative Filtering Logic)
# Step A: Only keep users who have rated more than 200 books
x = ratings['user_id'].value_counts() > 200
knowledgeable_users = x[x].index
filtered_ratings = ratings[ratings['user_id'].isin(knowledgeable_users)]

# Step B: Merge ratings with books on ISBN to get book titles
rating_with_books = filtered_ratings.merge(books, on='ISBN')

# Step C: Only keep books that have received at least 50 ratings
num_rating = rating_with_books.groupby('title')['rating'].count().reset_index()
num_rating.rename(columns={'rating': 'num_of_rating'}, inplace=True)

final_rating = rating_with_books.merge(num_rating, on='title')
final_rating = final_rating[final_rating['num_of_rating'] >= 50]

# Step D: Drop duplicate ratings by the same user for the same book
final_rating.drop_duplicates(['user_id', 'title'], inplace=True)

# 4. Create the Pivot Table
print("Building the rating matrix...")
book_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating')
book_pivot.fillna(0, inplace=True)

# 5. Model Training
print("Training the Nearest Neighbors model...")
book_sparse = csr_matrix(book_pivot)

model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

# 6. Recommendation Function
def recommend_book(book_name):
    # Find the index of the requested book
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
    except IndexError:
        print(f"\n[!] Book '{book_name}' not found in the trained dataset.")
        print("Note: The book might exist, but it may have been filtered out for having less than 50 ratings.")
        return
        
    # Get the nearest neighbors
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    
    print(f"\n📚 Top Recommendations for '{book_name}':")
    for i in range(len(suggestions[0])):
        if i != 0: # Skip the first one because it's the exact same book you searched for
            recommended_book = book_pivot.index[suggestions[0][i]]
            print(f" {i}. {recommended_book}")

# ==========================================
# 7. Test the Engine!
# ==========================================
print("\n--- Testing the Recommender Engine ---")
recommend_book("Harry Potter and the Chamber of Secrets (Book 2)")
recommend_book("Animal Farm")
# You can add more tests here later!