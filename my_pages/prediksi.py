import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.prediksi import prediksi_sentimen

def show():
    # Set judul aplikasi Streamlit
    st.title("Prediksi Sentimen Teks dalam Bahasa Indonesia")

    # Membuat layout dua kolom untuk pilihan input
    col1, col2 = st.columns(2)

    # Kolom kiri untuk input manual
    with col1:
        st.subheader("Input Manual")
        input_teks = st.text_area("Masukkan teks untuk prediksi:")
        
        if st.button('Prediksi Sentimen'):
            if input_teks:
                # Panggil fungsi prediksi_sentimen dari modul prediksi
                sentimen, prob = prediksi_sentimen(input_teks)
                
                # Tampilkan hasil prediksi
                st.write(f"Sentimen: {sentimen}")
                st.write(f"Probabilitas: {prob:.4f}")
                
                # Menyimpan hasil prediksi di session_state
                st.session_state.prediksi_sentimen_manual = {
                    "Sentimen": sentimen,
                    "Probabilitas": prob
                }

            else:
                st.write("Tolong masukkan teks terlebih dahulu.")

    # Kolom kanan untuk upload file CSV
    with col2:
        st.subheader("Upload File CSV")
        uploaded_file = st.file_uploader("Pilih file CSV untuk prediksi", type="csv")

        if uploaded_file is not None:
            try:
                # Membaca file CSV
                df = pd.read_csv(uploaded_file)
                
                # Identifikasi kolom dengan angka besar dan ubah menjadi string
                for col in df.select_dtypes(include=["int64", "float64"]).columns:
                    if df[col].max() > 2**53:  # Jika ada angka besar (melebihi 2^53)
                        df[col] = df[col].astype(str)
                
                # Menyimpan dataset di session_state
                st.session_state.uploaded_dataset = df
                st.session_state.dataset_analysis = {
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "columns_info": df.dtypes.to_dict()
                }

                # Menampilkan informasi dan preview data
                st.success("Dataset berhasil diunggah!")
                st.write("Preview Data CSV:")
                st.write(df.head())

                # Memastikan kolom 'full_text' ada dalam CSV
                if 'full_text' in df.columns:
                    # Menambahkan tombol untuk memulai prediksi
                    if st.button('Prediksi Sentimen untuk CSV'):
                        # Looping untuk setiap baris teks
                        results = []
                        for index, row in df.iterrows():
                            sentimen, prob = prediksi_sentimen(row['full_text'])  # Menggunakan 'full_text' untuk analisis
                            results.append({
                                "Teks": row['full_text'],
                                "Sentimen": sentimen,
                                "Probabilitas": prob
                            })

                        # Menampilkan hasil prediksi
                        result_df = pd.DataFrame(results)
                        st.write("Hasil Prediksi Sentimen:")
                        st.write(result_df)

                        # Menyimpan hasil prediksi CSV ke session_state
                        st.session_state.prediksi_sentimen_csv = result_df

                        # Visualisasi hasil prediksi untuk seluruh dataset
                        sentiment_counts = result_df['Sentimen'].value_counts()

                        # Membagi layout menjadi 2 kolom untuk visualisasi
                        col1, col2 = st.columns(2)

                        # Grafik Distribusi Sentimen pertama
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette='viridis')
                            ax.set_title('Distribusi Sentimen dari Dataset CSV')
                            ax.set_ylabel('Jumlah')
                            ax.set_xlabel('Sentimen')

                            # Menambahkan jumlah pasti di atas setiap bar
                            for i, count in enumerate(sentiment_counts.values):
                                ax.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=12)

                            st.pyplot(fig)

                else:
                    st.write("CSV tidak memiliki kolom 'full_text'. Pastikan CSV memiliki kolom yang berisi teks.")
            except Exception as e:
                st.error(f"Gagal memproses file CSV: {e}")

# Memanggil fungsi show() di bagian bawah jika perlu
if __name__ == "__main__":
    show()
