import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Buka file
df = pd.read_excel("evaluasi.xlsx")

# Tampilkan nama kolom yang sebenarnya
print("Kolom asli:", df.columns.tolist())

# Bersihkan nama kolom dari spasi dan karakter tak perlu
df.columns = df.columns.str.strip().str.replace(' ', '').str.replace('?', '', regex=False)

# Cek kembali nama kolom setelah dibersihkan
print("Kolom setelah dibersihkan:", df.columns.tolist())

# Ubah label menjadi angka (1 = relevant, 0 = not relevant)
df['label'] = df['Relevant'].apply(lambda x: 1 if str(x).strip().lower() == 'relevant' else 0)

# Hitung evaluasi untuk setiap query
results = []
for query in df['Query'].unique():
    subset = df[df['Query'] == query]
    y_true = subset['label']
    y_pred = [1] * len(subset)  # Asumsi semua hasil direkomendasikan

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results.append({
        'Query': query,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

# Simpan hasil ke Excel
eval_df = pd.DataFrame(results)
eval_df.to_excel("hasil_evaluasi.xlsx", index=False)
print("Hasil evaluasi disimpan dalam 'hasil_evaluasi.xlsx'")
