import nltk
from nltk import word_tokenize, pos_tag

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# User input
text = input("Enter legal text: ")

# Tokenize and POS tag
tokens = word_tokenize(text)
tags = pos_tag(tokens)

print("\nDetected Named Entities:")
count = 0

# Detect proper nouns as entities
for word, tag in tags:
    if tag == "NNP":
        print(word, "-> ENTITY")
        count += 1

# Calculate accuracy
actual = int(input("\nEnter actual number of entities: "))

if actual == 0 and count == 0:
    accuracy = 100
elif max(count, actual) == 0:
    accuracy = 0
else:
    accuracy = (min(count, actual) / max(count, actual)) * 100

print("\nPredicted Entities:", count)
print("NER Accuracy:", round(accuracy, 2), "%")