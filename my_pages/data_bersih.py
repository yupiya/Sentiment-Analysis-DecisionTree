import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    st.title("📊 Data Bersih - Cleaned Data dengan Sentimen Lokal")

    # Load the preprocessed dataset from local file
    st.write("📂 **Memuat Data Preprocessing Lokal**")
    try:
        # Baca file `sentimen_preprocessing.csv`
        preprocessed_data = pd.read_csv('sentimen_preprocessing.csv')

        st.write("✅ **Data Preprocessing Berhasil Dimuat**")
        st.dataframe(preprocessed_data.head())

        # Simpan preprocessed data ke session state
        st.session_state.tokenized_dataset = preprocessed_data

    except FileNotFoundError:
        st.error("🚨 File `sentimen_preprocessing.csv` tidak ditemukan. Pastikan file tersedia di direktori lokal.")
        return

    # Load sentiment analysis data from local file
    st.write("🧠 **Analisis Sentimen dari File Lokal**")
    try:
        # Baca file `sentimen_label_rework.csv`
        sentiment_data = pd.read_csv('sentimen_label_rework.csv')
        st.write("📂 **Data Sentimen Lokal Berhasil Dimuat**")
        st.dataframe(sentiment_data.head())

        # Merge the sentiment data with the preprocessed dataset
        merged_data = pd.merge(
            preprocessed_data,
            sentiment_data[['stemming', 'sentiment_score', 'sentiment']],
            on='stemming',
            how='inner'
        )

        # Hapus baris dengan nilai "none", "None", atau NaN di kolom `stemming`
        merged_data['stemming'] = merged_data['stemming'].str.strip()  # Hapus spasi di awal/akhir
        merged_data = merged_data[~merged_data['stemming'].str.lower().str.contains('none', na=False)]
        merged_data = merged_data.dropna(subset=['stemming']).reset_index(drop=True)

        # Hapus data pada indeks ke-108 jika ada
        if 108 in merged_data.index:
            merged_data = merged_data.drop(index=108).reset_index(drop=True)

        # Simpan hasil analisis sentimen ke session state
        st.session_state.sentiment_analysis = merged_data

        # Tampilkan hasil analisis sentimen
        st.write("🔎 **Hasil Analisis Sentimen**")
        st.dataframe(merged_data[['stemming', 'sentiment_score', 'sentiment']], use_container_width=True)

        # Create a button to download the data with sentiment analysis
        csv_with_sentiment = merged_data[['stemming', 'sentiment_score', 'sentiment']].to_csv(index=False)
        st.download_button(
            label="Download Data dengan Analisis Sentimen",
            data=csv_with_sentiment,
            file_name="data_dengan_sentimen.csv",
            mime="text/csv"
        )

        # Visualisasi jumlah masing-masing label sentimen
        st.write("📊 **Visualisasi Sentimen pada Data**")

        # Menghitung jumlah setiap label sentimen
        sentiment_counts = merged_data['sentiment'].value_counts()

        # Menampilkan angka pasti
        st.write(f"Jumlah data positif: {sentiment_counts.get('positif', 0)}")
        st.write(f"Jumlah data negatif: {sentiment_counts.get('negatif', 0)}")
        st.write(f"Jumlah data netral: {sentiment_counts.get('netral', 0)}")

        # Membuat dua kolom untuk visualisasi berdampingan
        col1, col2 = st.columns(2)

        # Visualisasi dengan menggunakan seaborn (Bar plot)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm', ax=ax)
            ax.set_title('Sentimen Data')
            ax.set_xlabel('Sentimen')
            ax.set_ylabel('Jumlah')
            st.pyplot(fig)

        # Visualisasi dengan menggunakan matplotlib (Pie chart)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=40, colors=sns.color_palette('coolwarm', n_colors=3))
            ax2.set_title('Distribusi Sentimen')
            st.pyplot(fig2)

    except FileNotFoundError:
        st.error("🚨 File `sentimen_label_rework.csv` tidak ditemukan. Pastikan file tersedia di direktori lokal.")
    except Exception as e:
        st.error(f"🚨 Terjadi kesalahan: {e}")
