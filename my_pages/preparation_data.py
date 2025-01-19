# tfidf_split_data.py
import streamlit as st
import pandas as pd
from utils.preparation_data import process_tfidf, split_data

def show():
    st.title("📊 Proses TF-IDF dan Pembagian Data")

    # Check if the tokenized dataset with sentiment analysis results is available
    if 'sentiment_analysis' not in st.session_state:
        st.warning("🚨 Data analisis sentimen belum tersedia. Silakan lengkapi langkah analisis sentimen terlebih dahulu.")
        return

    # Get the sentiment analysis data
    df = st.session_state.sentiment_analysis.copy()

    # Show some data for preview
    st.write("🗂️ **Data yang Akan Digunakan untuk TF-IDF**")
    st.dataframe(df.head(), use_container_width=True)

    # Process TF-IDF
    X, y, tvec = process_tfidf(df, text_column='stemming', label_column='sentiment')  # Assuming sentiment as label

    # Show TF-IDF feature shape
    st.write(f"🔢 **Shape dari TF-IDF**: {X.shape}")

    # Split data into train and test
    x_train, x_test, y_train, y_test = split_data(X, y)

    # Show the number of samples in each set
    st.write(f"💻 **Jumlah Data Training**: {x_train.shape[0]}")
    st.write(f"💻 **Jumlah Data Testing**: {x_test.shape[0]}")

    # Save results into session
    st.session_state.tfidf_split = {  # CONSISTENT KEY: tfidf_split
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'tvec': tvec  # Storing the TF-IDF vectorizer for later use
    }

    # Provide download buttons
    st.write("💾 **Unduh Data Training dan Testing**")
    
    # Convert the training and testing data to CSV format
    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)
    
    # Create download buttons
    st.download_button(
        label="Download Data Training",
        data=train_data.to_csv(index=False),
        file_name="data_training.csv",
        mime="text/csv"
    )
    
    st.download_button(
        label="Download Data Testing",
        data=test_data.to_csv(index=False),
        file_name="data_testing.csv",
        mime="text/csv"
    )
