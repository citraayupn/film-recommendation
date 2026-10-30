import pandas as pd
from gensim.models import KeyedVectors
from preprocess import simple_tokenize, parse_genres_column

class DatasetLoader:
    def __init__(self, path):
        self.path = path
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.path, on_bad_lines='skip', engine='python')

        if "top_1000" in self.path:
            self._process_top1000()
        elif "movies_metadata" in self.path:
            self._process_metadata()
        else:
            raise ValueError("❗ Format dataset tidak dikenali.")

        return self.df

    def _process_top1000(self):
        # Ambil kolom penting dan buang yang tidak diperlukan
        self.df.drop(columns=[
            "release_date", "original_language", "vote_average", "vote_count",
            "budget", "production_companies", "revenue", "runtime"
        ], inplace=True, errors='ignore')

        self.df = self.df[["title", "genres", "overview"]]
        self.df.dropna(subset=["title", "genres", "overview"], inplace=True)

        # Pastikan kolom genres dalam bentuk list
        self.df["genres"] = self.df["genres"].apply(lambda x: x if isinstance(x, list) else eval(x))

        # Tokenisasi overview
        self.df["overview"] = self.df["overview"].apply(simple_tokenize)

    def _process_metadata(self):
        # Ambil kolom penting
        self.df = self.df[["title", "genres", "overview"]]
        self.df.dropna(subset=["title", "genres", "overview"], inplace=True)

        # Parsing genres JSON string
        self.df["genres"] = self.df["genres"].apply(parse_genres_column)

        # Tokenisasi overview
        self.df["overview"] = self.df["overview"].apply(simple_tokenize)

# Fungsi pemuatan model Word2Vec
def load_word2vec_model(path="model/GoogleNews-vectors-negative300.bin"):
    return KeyedVectors.load_word2vec_format(path, binary=True, limit=300000)
