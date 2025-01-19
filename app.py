import streamlit as st
from streamlit_option_menu import option_menu

# Konfigurasi halaman utama Streamlit
st.set_page_config(page_title="Analisis Sentimen", layout="wide", initial_sidebar_state="expanded")

# Konfigurasi Sidebar Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Upload", "Preprocessing Data", "Data Bersih", "Preparation Data", "Decision Tree", "Prediksi"],  # Add Decision Tree
        icons=["house", "cloud-upload", "gear", "table", "wrench", "tree"],  # Icon for Decision Tree
        menu_icon="cast",
        default_index=0,
    )

# Navigasi ke file sesuai menu
if selected == "Home":
    from my_pages import home
    home.show()
elif selected == "Upload":
    from my_pages import upload
    upload.show()
elif selected == "Preprocessing Data":
    from my_pages import preprocessing
    preprocessing.show()
elif selected == "Data Bersih":  
    from my_pages import data_bersih
    data_bersih.show()
elif selected == "Preparation Data":
    from my_pages import preparation_data
    preparation_data.show()
elif selected == "Decision Tree":  
    from my_pages import decision_tree
    decision_tree.show()
elif selected == "Prediksi":  
    from my_pages import prediksi
    prediksi.show()
