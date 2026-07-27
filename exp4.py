import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Input number of documents
n = int(input("Enter number of documents: "))

# Input documents
documents = []
for i in range(n):
    doc = input(f"Enter document {i + 1}: ")
    documents.append(doc)

# Input query
query = input("\nEnter search query: ")

# Create TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert query to TF-IDF vector
query_vector = vectorizer.transform([query])

# Calculate TF-IDF cosine similarity
tfidf_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

print("\nTF-IDF Similarity Scores:")
for i, score in enumerate(tfidf_scores):
    print(f"Document {i + 1}: {score:.3f}")

# Find most relevant document using TF-IDF
best_tfidf = np.argmax(tfidf_scores)
print("\nMost Relevant Document (TF-IDF):")
print(documents[best_tfidf])

# ------------------ LSA ------------------

# Choose a safe number of components
n_components = min(2, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)

if n_components >= 1:
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    lsa_matrix = svd.fit_transform(tfidf_matrix)
    query_lsa = svd.transform(query_vector)

    lsa_scores = cosine_similarity(query_lsa, lsa_matrix)[0]

    print("\nLSA Similarity Scores:")
    for i, score in enumerate(lsa_scores):
        print(f"Document {i + 1}: {score:.3f}")

    best_lsa = np.argmax(lsa_scores)
    print("\nMost Relevant Document (LSA):")
    print(documents[best_lsa])
else:
    print("\nNot enough documents/features to perform LSA.")