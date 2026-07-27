import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist

# Download required NLTK data
nltk.download('punkt')

# User input
tweet = input("Enter a tweet: ")

# Tokenization
tokens = nltk.word_tokenize(tweet.lower())

print("\nTokens:")
print(tokens)

# Generate n-grams
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print("\nUnigrams:")
print(unigrams)

print("\nBigrams:")
print(bigrams)

print("\nTrigrams:")
print(trigrams)

# Word Frequency
fd = FreqDist(tokens)

print("\nWord Frequencies:")
for word, freq in fd.items():
    print(word, ":", freq)

# Sample HMM Prediction
print("\nHMM Prediction (Sample)")
print("AI -> NOUN")
print("improves -> VERB")
print("technology -> NOUN")