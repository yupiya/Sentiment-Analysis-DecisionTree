import streamlit as st
import pandas as pd
from utils.text_cleaning import clean_twitter_text
from utils.normalization import normalisasi
from utils.stopword_removal import stopword
from utils.tokenization import tokenizes
from utils.stemming import stemming

def show():
    st.title("⚙️ Preprocessing Data")

    # Check if the dataset has been uploaded
    if 'uploaded_dataset' not in st.session_state or st.session_state.uploaded_dataset is None:
        st.warning("🚨 Tidak ada dataset yang diunggah. Silakan unggah dataset di menu **Upload**.")
        return

    # Display the current dataset
    st.write("🔍 Dataset saat ini:")
    st.dataframe(st.session_state.uploaded_dataset, use_container_width=True, hide_index=True)

    # Define the tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Cleaning Data", "Normalization", "Stopword Removal", "Tokenization", "Stemming"])

    # Tab 1: Cleaning Data
    with tab1:
        st.subheader("🧹 Cleaning Data")
        df = st.session_state.uploaded_dataset.copy()

        if 'full_text' in df.columns:
            # Apply cleaning functions
            df['clean'] = df['full_text'].apply(clean_twitter_text)
            df['lowercase'] = df['clean'].str.lower()

            # Display five samples of each column with full content and no index
            st.write("🗂️ **Original Full Text (5 Samples)**")
            st.dataframe(df['full_text'].head(5), use_container_width=True, hide_index=True)

            st.write("🧹 **Cleaned Text (5 Samples)**")
            st.dataframe(df[['clean']].head(5), use_container_width=True, hide_index=True)

            st.write("🔤 **Lowercase Cleaned Text (5 Samples)**")
            st.dataframe(df[['lowercase']].head(5), use_container_width=True, hide_index=True)

            # Store the cleaned dataset in session state
            st.session_state.cleaned_dataset = df
            st.success("✅ Data berhasil dibersihkan!")
        else:
            st.warning("Kolom `full_text` tidak ditemukan dalam dataset.")

    # Tab 2: Normalization
    with tab2:
        st.subheader("🔠 Normalization of Text")
        
        if 'cleaned_dataset' in st.session_state:
            df = st.session_state.cleaned_dataset.copy()

            if 'lowercase' in df.columns:
                df['normalisasi'] = df['lowercase'].apply(lambda x: normalisasi(x))

                st.write("🔡 **Lowercase Text (5 Samples)**")
                st.dataframe(df[['lowercase']].head(5), use_container_width=True, hide_index=True)

                st.write("🔠 **Normalized Text (5 Samples)**")
                st.dataframe(df[['normalisasi']].head(5), use_container_width=True, hide_index=True)

                st.session_state.normalized_dataset = df
                st.success("✅ Data berhasil dinormalisasi!")
            else:
                st.warning("Kolom `lowercase` tidak ditemukan dalam dataset.")
        else:
            st.warning("Data yang telah dibersihkan belum disimpan. Silakan lanjutkan ke langkah pertama.")

    # Tab 3: Stopword Removal
    with tab3:
        st.subheader("🔤 Stopword Removal")
        
        if 'normalized_dataset' in st.session_state:
            df = st.session_state.normalized_dataset.copy()

            df['stopword'] = df['normalisasi'].apply(lambda x: stopword(x))

            st.write("🔠 **Normalized Text (5 Samples)**")
            st.dataframe(df[['normalisasi']].head(5), use_container_width=True, hide_index=True)

            st.write("🚫 **Stopword Removed Text (5 Samples)**")
            st.dataframe(df[['stopword']].head(5), use_container_width=True, hide_index=True)

            st.session_state.stopword_removed_dataset = df
            st.success("✅ Stopword telah berhasil dihapus!")
        else:
            st.warning("Data yang telah dinormalisasi belum tersedia. Silakan lanjutkan ke langkah kedua.")

    # Tab 4: Tokenization
    with tab4:
        st.subheader("📚 Tokenization")
        
        if 'stopword_removed_dataset' in st.session_state:
            df = st.session_state.stopword_removed_dataset.copy()

            df['tokenizes'] = df['stopword'].apply(lambda x: tokenizes(x))

            st.write("🔠 **Stopword Removed Text (5 Samples)**")
            st.dataframe(df[['stopword']].head(5), use_container_width=True, hide_index=True)

            st.write("✂️ **Tokenized Text (5 Samples)**")
            st.dataframe(df[['tokenizes']].head(5), use_container_width=True, hide_index=True)

            st.session_state.tokenized_dataset = df
            st.success("✅ Data berhasil ditokenisasi!")
        else:
            st.warning("Data dengan stopword yang telah dihapus belum tersedia. Silakan lanjutkan ke langkah sebelumnya.")

    # Tab 5: Stemming from CSV
    with tab5:
        st.subheader("🌿 Stemming")
        
        # Show Tokenized Text first
        if 'tokenized_dataset' in st.session_state:
            df = st.session_state.tokenized_dataset.copy()

            st.write("✂️ **Tokenized Text (5 Samples)**")
            st.dataframe(df[['tokenizes']].head(5), use_container_width=True, hide_index=True)

            # Load the CSV file and show the 'stemming' column
            try:
                # Read the CSV file to get the stemming column
                file_path = 'sentimen_preprocessing.csv'  # Assuming it's in the same directory
                df_stemming = pd.read_csv(file_path)

                # Check if 'stemming' column exists
                if 'stemming' in df_stemming.columns:
                    st.write("🌱 **Stemmed Text (5 Samples)**")
                    st.dataframe(df_stemming[['stemming']].head(5), use_container_width=True, hide_index=True)

                    # Add stemming results to the tokenized dataset
                    df['stemming'] = df_stemming['stemming']

                    # Save the updated DataFrame to session state
                    st.session_state.tokenized_dataset = df
                    st.success("✅ Data stemming berhasil dimuat dari file dan disimpan!")
                else:
                    st.warning("Kolom `stemming` tidak ditemukan dalam file CSV.")
            
            except FileNotFoundError:
                st.error(f"🚨 File `{file_path}` tidak ditemukan. Pastikan file CSV tersedia.")
            except Exception as e:
                st.error(f"🚨 Terjadi kesalahan: {str(e)}")
