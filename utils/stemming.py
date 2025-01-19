# utils/stemming.py

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Fungsi stemming
def stemming(tokenized_text):
    """Function to perform stemming on tokenized text."""
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_words = [stemmer.stem(word) for word in tokenized_text]
    return " ".join(stemmed_words)
