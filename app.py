from model_loader import DatasetLoader, load_word2vec_model
from recommender import recommend
from eda import show_genre_distribution, show_overview_length_distribution, show_wordcloud
from visualizer import show_recommendation_chart, show_tsne_plot
from tabulate import tabulate
import time

# Fungsi untuk memotong teks panjang agar tabel tidak rusak
def crop_text(text, max_len=40):
    return text if len(text) <= max_len else text[:max_len-3] + "..."

# Memuat model Word2Vec
print("🔄 Memuat model Word2Vec...")
wv = load_word2vec_model("model/GoogleNews-vectors-negative300.bin")

# Memilih dataset
print("\n📂 Pilih dataset yang akan digunakan:")
print("1. Dataset 1 (top_1000_popular_movies_tmdb.csv)")
print("2. Dataset 2 (movies_metadata.csv)")
pilihan = input("Masukkan angka [1/2]: ").strip()

if pilihan == "1":
    dataset_path = "data/top_1000_popular_movies_tmdb.csv"
elif pilihan == "2":
    dataset_path = "data/movies_metadata.csv"
else:
    print("❗ Pilihan tidak valid. Keluar.")
    exit()

# Memuat dataset
print(f"\n🔄 Memuat dataset film dari {dataset_path}...")
loader = DatasetLoader(dataset_path)
df = loader.load()

# Menampilkan EDA
print("📊 Menampilkan analisis eksploratif (EDA)...")
show_genre_distribution(df)
show_overview_length_distribution(df)
show_wordcloud(df)

# Visualisasi awal t-SNE seluruh dataset
show_tsne_plot(df, wv)

print("\n✅ Sistem siap digunakan.")
print("Ketik sinopsis atau deskripsi film (ketik 'exit' untuk keluar).")

# Loop interaktif
while True:
    query = input("\n📥 Masukkan sinopsis film: ")
    if query.lower() in ["exit", "keluar", "quit"]:
        print("🚪 Keluar dari sistem rekomendasi.")
        break

    print(f"\n🔎 Anda mencari film dengan makna serupa: '{query}'")

    start_time = time.time()
    results = recommend(query, df, wv)
    execution_time = time.time() - start_time

    if results.empty:
        print("❗Tidak ditemukan rekomendasi yang cukup mirip berdasarkan input Anda.")
    else:
        results["genres"] = results["genres"].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        results["similarity_score_float"] = results["similarity_score"]
        results["similarity_score"] = results["similarity_score"].apply(lambda x: f"{x:.3f}")

        results_display = results.copy()
        results_display["title"] = results_display["title"].apply(lambda x: crop_text(str(x)))
        results_display["genres"] = results_display["genres"].apply(lambda x: crop_text(str(x)))

        print("\n🎯 Rekomendasi Film Berdasarkan Kemiripan Makna:")
        print(tabulate(
            results_display[["title", "genres", "similarity_score"]],
            headers=["Judul Film", "Genre", "Skor Kemiripan"],
            tablefmt="fancy_grid",
            showindex=False,
            stralign="left",
            numalign="center"
        ))

        # Visualisasi hasil rekomendasi
        show_recommendation_chart(results)

        # Visualisasi t-SNE ulang (berdasarkan seluruh dataset)
        show_tsne_plot(df, wv)

        # Info tambahan
        print(f"\n📊 Jumlah data dalam dataset: {len(df)} film")
        print(f"⏱️ Waktu eksekusi rekomendasi: {execution_time:.3f} detik")
