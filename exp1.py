import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Sample text
text = """
Natural Language Processing is an important field of Artificial Intelligence.
It helps computers understand human languages and process information efficiently.
Students are studying various machine learning techniques for text analysis.
Researchers are running experiments and analyzing large datasets.
The children were playing games and enjoying their holidays.
Stemming reduces words to their root forms while lemmatization produces meaningful base words.
"""

# Sentence Tokenization
sentences = sent_tokenize(text)

print("\nORIGINAL TEXT")
print("-" * 60)
print(text)

print("\nSENTENCE TOKENS")
print("-" * 60)
for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")

# Initialize Stemmer and Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Process each sentence
for i, sentence in enumerate(sentences, 1):

    print("\n" + "=" * 60)
    print(f"Sentence {i}")
    print("=" * 60)

    # Word Tokenization
    words = word_tokenize(sentence)

    # Stemming
    stemmed_words = [stemmer.stem(word) for word in words]

    # Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    print("\nOriginal Sentence:")
    print(sentence)

    print("\nWord Tokens:")
    print(words)

    print("\nStemmed Words:")
    print(stemmed_words)

    print("\nLemmatized Words:")
    print(lemmatized_words)

# Stemming vs Lemmatization Comparison
print("\n")
print("=" * 60)
print("STEMMING VS LEMMATIZATION")
print("=" * 60)

print(f"{'Original':<15}{'Stemmed':<15}{'Lemmatized':<15}")
print("-" * 45)

for sentence in sentences:
    words = word_tokenize(sentence)

    for word in words:
        if word.isalpha():  # Ignore punctuation
            stemmed = stemmer.stem(word)
            lemmatized = lemmatizer.lemmatize(word)

            print(f"{word:<15}{stemmed:<15}{lemmatized:<15}")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print("1. Tokenization splits text into sentences and words.")
print("2. Stemming removes prefixes/suffixes and may create non-dictionary words.")
print("3. Lemmatization produces meaningful dictionary base forms.")
print("4. Lemmatization is generally more accurate for NLP and sentiment analysis.")