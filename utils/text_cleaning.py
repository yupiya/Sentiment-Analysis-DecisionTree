import re
import pandas as pd

def clean_twitter_text(text):
    # Menghapus mention (@username)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Menghapus hashtag (#hashtag)
    text = re.sub(r'#\w+', '', text)
    # Menghapus retweet (RT)
    text = re.sub(r'RT[\s]+', '', text)
    # Menghapus URL
    text = re.sub(r'https?://\S+', '', text)
    # Menghapus karakter selain huruf dan angka
    text = re.sub(r'[^A-Za-z0-9 ]', '', text)
    # Menghapus spasi lebih dari satu dan menghapus spasi di awal/akhir teks
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(df):
    # Terapkan fungsi pembersihan pada kolom 'full_text'
    df['clean'] = df['full_text'].apply(clean_twitter_text)
    # Konversi teks menjadi lowercase
    df['lowercase'] = df['clean'].str.lower()

    # Hapus duplikat berdasarkan kolom 'lowercase'
    df = df.drop_duplicates(subset=['lowercase'], keep='first').reset_index(drop=True)

    return df
