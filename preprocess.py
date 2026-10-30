import re
import json

STOPWORDS = set([
    "a", "an", "the", "and", "or", "in", "on", "at", "for", "with", "without", "to", "of",
    "from", "is", "are", "was", "were", "be", "been", "it", "that", "this", "by", "as", "but", 
    "if", "not", "into", "about", "after", "before", "over", "under", "then"
])

def simple_tokenize(text):
    """Tokenisasi dasar + stopword removal"""
    if isinstance(text, str):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [w for w in tokens if w not in STOPWORDS]
    return []

def parse_genres_column(genres_raw):
    """
    Mengubah string JSON genres menjadi list nama genre.
    Jika kosong atau gagal diparse, kembalikan ['Unknown'].
    """
    try:
        if isinstance(genres_raw, str) and genres_raw.strip():
            genres_list = json.loads(genres_raw.replace("'", '"'))
            names = [g["name"] for g in genres_list if isinstance(g, dict) and "name" in g]
            return names if names else ["Unknown"]
    except Exception:
        pass
    return ["Unknown"]

