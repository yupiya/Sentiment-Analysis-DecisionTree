import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.decision_tree import train_decision_tree  # Ensure this import is correct

def show():
    st.title("🌳 Decision Tree")
    st.write("Klasifikasi menggunakan Decision Tree dengan parameter manual.")

    # Pastikan data TF-IDF tersedia
    if 'tfidf_split' not in st.session_state:  # CONSISTENT KEY: tfidf_split
        st.warning("🚨 Data hasil TF-IDF dan split belum tersedia. Silakan lakukan preprocessing pada langkah sebelumnya.")
        return

    # Load data TF-IDF dari session state
    x_train = st.session_state.tfidf_split['x_train']
    y_train = st.session_state.tfidf_split['y_train']
    x_test = st.session_state.tfidf_split['x_test']
    y_test = st.session_state.tfidf_split['y_test']

    # Training dan evaluasi Decision Tree
    st.write("🔄 **Training Model dengan Parameter Manual**")
    with st.spinner("Sedang melatih model..."):
        results = train_decision_tree(x_train, y_train, x_test, y_test)

    # Tampilkan parameter yang digunakan
    st.write("✅ **Parameter yang Digunakan**")
    st.json(results["params_used"])

    # Tampilkan metrik evaluasi
    st.write("📊 **Hasil Evaluasi Model**")
    st.write(f"- **Akurasi**: {results['accuracy']:.2f}%")
    st.write(f"- **Precision**: {results['precision']:.2f}%")
    st.write(f"- **Recall**: {results['recall']:.2f}%")
    st.write(f"- **F1 Score**: {results['f1_score']:.2f}%")

    # Visualisasi confusion matrix
    st.write("🟦 **Visualisasi Confusion Matrix**")
    conf_matrix = results["confusion_matrix"]
    
    # Membuat dua kolom untuk menempatkan gambar di sebelah kiri
    col1, col2 = st.columns([1, 2])  # Kolom pertama lebih kecil, kolom kedua lebih besar

    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))  # Ukuran gambar lebih kecil
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        st.pyplot(fig)

    # Tampilkan classification report
    st.write("📜 **Classification Report**")
    st.text(results["classification_report"])

    # Simpan hasil ke session_state
    st.session_state.decision_tree_results = results
