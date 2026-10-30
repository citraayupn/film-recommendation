import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def show_genre_distribution(df):
    """
    Menampilkan distribusi genre film (bar chart),
    dengan kategori 'Unknown' untuk genre kosong.
    """
    # Ubah genre kosong [] menjadi ['Unknown']
    df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else ['Unknown'])

    # Pecah semua genre menjadi baris tunggal
    all_genres = df['genres'].explode()

    # Hitung jumlah tiap genre
    genre_counts = all_genres.value_counts().head(10)

    # Plot
    plt.figure(figsize=(10, 6))
    genre_counts.plot(kind='bar', color='cornflowerblue')
    plt.title('Distribusi Genre Film')
    plt.xlabel('Genre')
    plt.ylabel('Jumlah Film')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def show_overview_length_distribution(df):
    """
    Menampilkan histogram panjang sinopsis (jumlah kata).
    """
    df['overview_length'] = df['overview'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['overview_length'], bins=30, kde=True, color="mediumseagreen")
    plt.title('Distribusi Panjang Sinopsis')
    plt.xlabel('Jumlah Kata')
    plt.ylabel('Jumlah Film')
    plt.tight_layout()
    plt.show()

def show_wordcloud(df):
    """
    Menampilkan word cloud dari semua sinopsis film (setelah tokenisasi).
    """
    text = ' '.join([' '.join(overview) for overview in df['overview'] if isinstance(overview, list)])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("WordCloud Kata Umum di Sinopsis Film")
    plt.tight_layout()
    plt.show()
