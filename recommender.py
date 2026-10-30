import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import simple_tokenize

def vectorize(tokens, wv):
    vectors = [wv[w] for w in tokens if w in wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

def recommend(input_text, df, wv, top_n=10):
    input_tokens = simple_tokenize(input_text)
    input_vec = vectorize(input_tokens, wv).reshape(1, -1)

    doc_matrix = np.vstack([vectorize(row, wv) for row in df["overview"]])
    sims = cosine_similarity(input_vec, doc_matrix)[0]

    df_result = df.copy()
    df_result["similarity_score"] = sims
    return df_result.sort_values(by="similarity_score", ascending=False)[["title", "genres", "similarity_score"]].head(top_n)

# skenario pake fold cross validat ditulis brp foldr pake sekian fol itu hasilnya gmn 
# fold cross valid /rasio pembagian daata
# baginya 80/20 nnti muncul cross va... brp