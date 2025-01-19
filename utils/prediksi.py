import torch
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis")

# Fungsi untuk memprediksi sentimen teks
def prediksi_sentimen(teks):
    # Menggunakan pipeline untuk mendapatkan hasil sentimen
    result = pipe(teks)
    
    # Hasil prediksi berupa list, ambil label dan skor probabilitas
    sentimen = result[0]['label']
    prob = result[0]['score']
    
    # Tampilkan hasil
    return sentimen, prob

# Contoh penggunaan
input_teks = "Terima kasih atas bantuannya ya!"
sentimen, prob = prediksi_sentimen(input_teks)

print(f"Sentimen: {sentimen}, Probabilitas: {prob:.4f}")
