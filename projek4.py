import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Rekomendasi Wisata Indonesia",
    page_icon="🏝️",
    layout="wide"
)

@st.cache_data
def load_data():
    df_places = pd.read_csv('tourism_with_id.csv')
    df_ratings = pd.read_csv('tourism_rating.csv')
    df_users = pd.read_csv('user.csv')
    df_all = pd.merge(df_ratings, df_places[['Place_Id', 'Place_Name', 'Category', 'City']], on='Place_Id')
    
    return df_places, df_ratings, df_users, df_all

try:
    df_places, df_ratings, df_users, df_all = load_data()
    all_place_names = sorted(df_places['Place_Name'].unique().tolist())
except FileNotFoundError:
    st.error("File CSV tidak ditemukan. Pastikan file 'tourism_with_id.csv', 'tourism_rating.csv', dan 'user.csv' ada di direktori yang sama.")
    st.stop()


if 'new_user_ratings' not in st.session_state:
    st.session_state.new_user_ratings = {}

# Header
st.title("🏝️ Sistem Rekomendasi Wisata Indonesia")
st.markdown("### Menggunakan Data Real & User-Based Collaborative Filtering")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
num_recommendations = st.sidebar.slider("Jumlah rekomendasi:", 5, 20, 10)
similarity_threshold = st.sidebar.slider("Threshold kemiripan:", 0.05, 0.9, 0.1, 0.05)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Rekomendasi", "⭐ Input Rating", "📊 Statistik Data"])

# --- TAB 1: REKOMENDASI ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        method = st.radio("Pilih Tipe User:", ["User Baru (Input Sendiri)", "User Terdaftar (ID)"])
        
        target_user_id = None
        current_ratings_display = []

        if method == "User Baru (Input Sendiri)":
            if st.session_state.new_user_ratings:
                st.success(f"✅ {len(st.session_state.new_user_ratings)} rating tersimpan.")
                target_user_id = 9999
            else:
                st.warning("Silakan isi rating di Tab ⭐ dulu.")
        else:
            selected_id = st.selectbox("Pilih User ID dari Database:", df_users['User_Id'].unique())
            target_user_id = selected_id
            user_data = df_all[df_all['User_Id'] == selected_id].head(5)
            st.write("**Beberapa tempat yang disukai user ini:**")
            for _, row in user_data.iterrows():
                st.write(f"- {row['Place_Name']} ({row['Place_Ratings']}⭐)")

    with col2:
        if st.button("🚀 Hitung Rekomendasi", type="primary", use_container_width=True):
            with st.spinner("Menganalisis kemiripan antar user..."):
                
                # Siapkan data untuk matrix
                if method == "User Baru (Input Sendiri)":
                    new_data = pd.DataFrame([
                        {'User_Id': 9999, 'Place_Name': k, 'Place_Ratings': v} 
                        for k, v in st.session_state.new_user_ratings.items()
                    ])
                    # Kita butuh Place_Id untuk konsistensi matrix
                    new_data = pd.merge(new_data, df_places[['Place_Id', 'Place_Name']], on='Place_Name')
                    processing_df = pd.concat([df_all[['User_Id', 'Place_Id', 'Place_Ratings']], 
                                             new_data[['User_Id', 'Place_Id', 'Place_Ratings']]])
                else:
                    processing_df = df_all

                # Create User-Item Matrix
                matrix = processing_df.pivot_table(index='User_Id', columns='Place_Id', values='Place_Ratings').fillna(0)
                
                # Cosine Similarity
                user_sim = cosine_similarity(matrix)
                user_sim_df = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)
                
                # Ambil user serupa
                similar_users = user_sim_df[target_user_id].sort_values(ascending=False)
                similar_users = similar_users[(similar_users > similarity_threshold) & (similar_users.index != target_user_id)]
                
                if similar_users.empty:
                    st.error("Tidak ditemukan user dengan minat serupa. Coba turunkan threshold kemiripan.")
                else:
                    # Prediksi Rating
                    user_visited = processing_df[processing_df['User_Id'] == target_user_id]['Place_Id'].tolist()
                    unvisited_places = df_places[~df_places['Place_Id'].isin(user_visited)]['Place_Id']
                    
                    recommendations = []
                    for p_id in unvisited_places:
                        weighted_sum = 0
                        sim_sum = 0
                        for sim_user, score in similar_users.items():
                            rating = matrix.loc[sim_user, p_id]
                            if rating > 0:
                                weighted_sum += score * rating
                                sim_sum += score
                        
                        if sim_sum > 0:
                            pred = weighted_sum / sim_sum
                            recommendations.append({'Place_Id': p_id, 'Prediksi': pred})
                    
                    # Tampilkan Hasil
                    rec_df = pd.DataFrame(recommendations).sort_values('Prediksi', ascending=False).head(num_recommendations)
                    rec_df = pd.merge(rec_df, df_places[['Place_Id', 'Place_Name', 'City', 'Category']], on='Place_Id')
                    
                    st.success(f"Ditemukan {len(rec_df)} rekomendasi untuk Anda!")
                    for _, row in rec_df.iterrows():
                        with st.expander(f"📍 {row['Place_Name']} ({row['City']})"):
                            st.write(f"**Kategori:** {row['Category']}")
                            st.write(f"**Prediksi Skor Kepuasan:** {row['Prediksi']:.2f}/5.0")
                            st.progress(row['Prediksi']/5)

# --- TAB 2: INPUT RATING ---
with tab2:
    st.header("⭐ Beri Rating Pengalaman Anda")
    selected_tour = st.selectbox("Cari Tempat Wisata:", all_place_names)
    rating_val = st.select_slider("Rating Anda:", options=[1, 2, 3, 4, 5], value=3)
    
    if st.button("Simpan Rating"):
        st.session_state.new_user_ratings[selected_tour] = rating_val
        st.toast(f"Berhasil menyimpan rating untuk {selected_tour}!")

    if st.session_state.new_user_ratings:
        st.write("---")
        st.write("**Rating yang telah Anda masukkan:**")
        for p, r in st.session_state.new_user_ratings.items():
            st.write(f"✅ {p}: {r} ⭐")

with tab3:
    st.header("📊 Statistik Destinasi")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Destinasi", len(df_places))
    c2.metric("Total User", len(df_users))
    c3.metric("Total Rating", len(df_ratings))
    
    st.subheader("Top 10 Destinasi Terpopuler")
    populer = df_all.groupby('Place_Name')['Place_Ratings'].count().sort_values(ascending=False).head(10)
    st.bar_chart(populer)
