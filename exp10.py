from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# List to store posts
posts = []

# Input number of posts
n = int(input("Enter number of posts: "))

# Input posts
for i in range(n):
    post = input(f"Enter post {i + 1}: ")
    posts.append(post)

# Input number of clusters
k = int(input("Enter number of clusters: "))

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(posts)

# Apply K-Means Clustering
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

model.fit(X)

# Get cluster labels
labels = model.labels_

print("\nCluster Results:\n")

for i in range(len(posts)):
    print("Post:", posts[i])
    print("Cluster:", labels[i])
    print()

# Display important keywords for each cluster
terms = vectorizer.get_feature_names_out()

print("Important Keywords:\n")

for i in range(k):
    center = model.cluster_centers_[i]
    top = center.argsort()[-5:]

    print("Cluster", i)
    for j in top:
        print(terms[j])
    print()

# Marketing Insight
print("Marketing Insight:")
print("Similar customer opinions are grouped together.")
print("Clusters help identify product trends and issues.")