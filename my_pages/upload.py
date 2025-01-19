import streamlit as st
import pandas as pd

def show():
    st.title("📤 Upload Dataset Mentah")

    if 'uploaded_dataset' not in st.session_state:
        st.session_state.uploaded_dataset = None
        st.session_state.dataset_analysis = None  # Menyimpan analisis dataset

    # Komponen upload file
    uploaded_file = st.file_uploader("Unggah file CSV Anda di sini", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Identifikasi kolom dengan angka besar dan ubah menjadi string
            for col in df.select_dtypes(include=["int64", "float64"]).columns:
                if df[col].max() > 2**53:
                    df[col] = df[col].astype(str)
            
            st.session_state.uploaded_dataset = df
            st.session_state.dataset_analysis = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "columns_info": df.dtypes.to_dict()
            }
            st.success("Dataset berhasil diunggah!")
            
            # Opsi tampilan tabel
            st.write("📊 **Pratinjau Data:**")
            num_rows = st.slider("Pilih jumlah baris untuk ditampilkan:", min_value=5, max_value=100, value=10)
            
            # Tampilkan tabel dengan scroll dan penggunaan lebar kontainer penuh
            st.dataframe(df.head(num_rows), use_container_width=True)
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
    
    elif st.session_state.uploaded_dataset is not None:
        df = st.session_state.uploaded_dataset
        st.write("📂 Dataset yang sebelumnya diunggah:")
        st.dataframe(df, use_container_width=True)
    
    else:
        st.info("Silakan unggah file CSV untuk melihat pratinjau.")

    # Menampilkan analisis dataset
    if st.session_state.dataset_analysis:
        st.write("📊 **Statistik Dataset:**")
        st.write(f"- **Jumlah Baris:** {st.session_state.dataset_analysis['rows']}")
        st.write(f"- **Jumlah Kolom:** {st.session_state.dataset_analysis['columns']}")
        st.write("📋 **Informasi Kolom**:")
        st.write(st.session_state.dataset_analysis['columns_info'])
