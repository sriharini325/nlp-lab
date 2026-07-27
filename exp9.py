from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Lists to store documents and labels
docs = []
labels = []

# Input number of documents
n = int(input("Enter number of documents: "))

# Input documents and categories
for i in range(n):
    docs.append(input(f"Enter document {i + 1}: "))
    labels.append(input(f"Enter category {i + 1}: "))

# Rule-Based Prediction
rule_pred = []

for doc in docs:
    doc = doc.lower()
    if "contract" in doc:
        rule_pred.append("contract")
    elif "judgment" in doc:
        rule_pred.append("judgment")
    else:
        rule_pred.append("agreement")

# Rule-Based Accuracy
rule_acc = accuracy_score(labels, rule_pred)

# Maximum Entropy (Logistic Regression)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

ml_pred = model.predict(X)
ml_acc = accuracy_score(labels, ml_pred)

# Display Results
print("\nRule-Based Accuracy:", round(rule_acc, 2))
print("Maximum Entropy Accuracy:", round(ml_acc, 2))