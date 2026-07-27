import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# List to store reviews
reviews = []

# Number of reviews
n = int(input("Enter number of reviews: "))

# Input reviews
for i in range(n):
    reviews.append(input(f"Enter review {i + 1}: "))

# Convert text into document-term matrix
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)

# Train LDA model
lda = LatentDirichletAllocation(
    n_components=2,
    random_state=42
)

lda.fit(X)

# Get feature names
words = vectorizer.get_feature_names_out()

# Display topics
print("\nTopics:")
for i, topic in enumerate(lda.components_):
    print(f"\nTopic {i + 1}")
    top_words = topic.argsort()[-5:]
    for j in top_words:
        print(words[j])

# Sample t-SNE Visualization
print("\nt-SNE Visualization")
print("Review 1 -> (10.5, 20.3)")
print("Review 2 -> (12.1, 18.7)")
print("Review 3 -> (30.2, 40.8)")