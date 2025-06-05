import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# --- Cargar datos ---
@st.cache_data
def load_data():
    df = pd.read_pickle(r'C:\Users\Melissa\Documents\Maestria Ciencia Datos\Vision_Computador\3-ViT & Transferencia\caltech256_features_mobilenet.pkl')
    df['clean_category'] = df['category'].apply(lambda x: x.split('.')[-1].replace('-', ' ').lower())
    return df

df = load_data()

# --- Función de búsqueda ---
def search_images(query, top_k):
    query = query.lower()
    matches = df[df['clean_category'].str.contains(query)]

    if matches.empty:
        return []

    query_vec = np.mean(matches['features'].to_list(), axis=0).reshape(1, -1)
    all_vecs = np.vstack(df['features'].to_list())
    sims = cosine_similarity(query_vec, all_vecs).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    return df.iloc[top_indices]

# --- Interfaz Streamlit ---
st.title("🔍 Búsqueda de imágenes por texto - Caltech256")

query = st.text_input("Introduce tu consulta (ej. 'helicopter', 'guitar')", value="guitar")
top_k = st.slider("Número de imágenes a mostrar", min_value=1, max_value=10, value=5)

if st.button("Buscar"):
    results = search_images(query, top_k)
    
    if results.empty:
        st.warning("No se encontraron imágenes para esa consulta.")
    else:
        st.success(f"{len(results)} imágenes encontradas.")
        for i, row in results.iterrows():
            st.image(Image.open(row['image_path']), caption=row['clean_category'], use_column_width=True)
