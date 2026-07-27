import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Get paragraph input from user
text = input("Enter a paragraph:\n")

# Tokenize paragraph into words
tokens = word_tokenize(text)

# Perform POS tagging
tagged_words = pos_tag(tokens)

# Display tokens
print("\nTOKENS:")
print(tokens)

# Display POS tags
print("\nPOS TAGS:")
for word, tag in tagged_words:
    print(f"{word} -> {tag}")

# Count tagged words
print("\nTotal Words:", len(tokens))