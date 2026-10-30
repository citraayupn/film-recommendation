import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.cm as cm

def wrap_label(label, width=35):
    """
    Membungkus teks panjang agar tidak melebar.
    """
    return '\n'.join(textwrap.wrap(label, width))

def show_recommendation_chart(results):
    """
    Menampilkan bar chart horizontal dari hasil rekomendasi film.
    """
    sorted_results = results.sort_values(by="similarity_score_float", ascending=False).head(10)

    # Bungkus label judul panjang agar rapi
    sorted_results["wrapped_title"] = sorted_results["title"].apply(lambda x: wrap_label(str(x), width=35))

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=sorted_results,
        x="similarity_score_float",
        y="wrapped_title",
        palette="viridis"
    )

    for i, (score, label) in enumerate(zip(sorted_results["similarity_score_float"], sorted_results["wrapped_title"])):
        ax.text(score + 0.01, i, f"{score:.3f}", va='center', fontsize=9)

    plt.xlabel("Skor Kemiripan")
    plt.ylabel("Judul Film")
    plt.title("🎯 Visualisasi Rekomendasi Film Berdasarkan Kemiripan", fontsize=13, pad=15)
    plt.tight_layout()
    plt.show()

def show_tsne_plot(df, wv, sample_size=300):
    """
    Menampilkan visualisasi t-SNE dari representasi vektor sinopsis film.
    Setiap titik diberi warna berbeda dan label judul di samping titik.
    """
    print("📉 Menyiapkan visualisasi t-SNE...")

    sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    overviews = sampled_df["overview"]
    titles = sampled_df["title"].apply(str).tolist()

    # Ubah sinopsis menjadi vektor Word2Vec
    vectors = np.vstack([
        np.mean([wv[w] for w in tokens if w in wv] or [np.zeros(300)], axis=0)
        for tokens in overviews
    ])

    # Reduksi dimensi ke 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(vectors)

    plt.figure(figsize=(14, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(titles)))

    for i, (x, y) in enumerate(reduced):
        plt.scatter(x, y, color=colors[i], s=30, alpha=0.7)
        plt.text(x + 1.2, y, titles[i], fontsize=8, alpha=0.8)

    plt.title("t-SNE Visualisasi Vektor Sinopsis Film")
    plt.xlabel("Dimensi 1")
    plt.ylabel("Dimensi 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


