import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman
st.set_page_config(page_title="Sistem Rekomendasi Pariwisata", layout="wide")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    df_tourism = pd.read_csv('tourism_with_id.csv')
    df_rating = pd.read_csv('tourism_rating.csv')
    df_user = pd.read_csv('user.csv')
    # Membersihkan data dan menghapus kolom kosong jika ada
    df_tourism = df_tourism.drop(columns=['Unnamed: 11', 'Unnamed: 12'], errors='ignore')
    return df_tourism, df_rating, df_user

# Load data
df_tourism, df_rating, df_user = load_data()

# --- PREPROCESSING ---
@st.cache_resource
def prepare_content_based(df):
    # Menggabungkan fitur untuk kesamaan konten
    df['content'] = df['Category'] + " " + df['Description'] + " " + df['City']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = prepare_content_based(df_tourism)

# --- FUNGSI REKOMENDASI ---

def get_recommendations_by_item(title, cosine_sim=cosine_sim):
    idx = df_tourism[df_tourism['Place_Name'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6] 
    item_indices = [i[0] for i in sim_scores]
    return df_tourism.iloc[item_indices]

def get_popular_recommendations(city='All', n=5):
    if city == 'All':
        return df_tourism.sort_values(by='Rating', ascending=False).head(n)
    else:
        return df_tourism[df_tourism['City'] == city].sort_values(by='Rating', ascending=False).head(n)

# --- UI STREAMLIT ---

st.title("🌴 Pesona Indonesia: Rekomendasi Pariwisata")

menu = st.sidebar.selectbox("Pilih Halaman", ["Home", "Cari Serupa", "Rekomendasi User", "Eksplorasi Kota"])

if menu == "Home":
    st.subheader("🔥 Destinasi Terpopuler")
    city_filter = st.selectbox("Filter berdasarkan Kota:", ["All"] + list(df_tourism['City'].unique()))
    popular_df = get_popular_recommendations(city_filter)
    
    cols = st.columns(3)
    for i, (idx, row) in enumerate(popular_df.iterrows()):
        with cols[i % 3]:
            st.info(f"**{row['Place_Name']}**")
            st.write(f"📍 {row['City']} | ⭐ {row['Rating']}")
            st.write(f"💰 Rp {row['Price']:,}")

elif menu == "Cari Serupa":
    st.subheader("🔍 Cari Tempat Wisata Serupa")
    selected_place = st.selectbox("Pilih tempat yang Anda sukai:", df_tourism['Place_Name'].tolist())
    
    if st.button("Tampilkan Rekomendasi"):
        recs = get_recommendations_by_item(selected_place)
        st.write(f"Berdasarkan **{selected_place}**, kami merekomendasikan:")
        for _, row in recs.iterrows():
            with st.expander(f"{row['Place_Name']} - {row['City']}"):
                st.write(f"**Kategori:** {row['Category']}")
                st.write(f"**Rating:** ⭐ {row['Rating']}")
                st.write(row['Description'])

elif menu == "Rekomendasi User":
    st.subheader("👤 Rekomendasi Personal")
    user_id = st.number_input("Masukkan User ID (1-300):", min_value=1, max_value=300, value=1)
    
    user_ratings = df_rating[df_rating['User_Id'] == user_id].merge(df_tourism, on='Place_Id')
    st.write(f"Riwayat kunjungan/rating User {user_id}:")
    st.dataframe(user_ratings[['Place_Name', 'Place_Ratings', 'Category', 'City']].sort_values(by='Place_Ratings', ascending=False).head(5))

elif menu == "Eksplorasi Kota":
    st.subheader("🗺️ Eksplorasi Data Wisata per Kota")
    selected_city = st.multiselect("Pilih Kota:", df_tourism['City'].unique(), default=df_tourism['City'].unique()[0])
    
    filtered_df = df_tourism[df_tourism['City'].isin(selected_city)]
    st.write(f"Menampilkan {len(filtered_df)} destinasi.")
    st.dataframe(filtered_df[['Place_Name', 'Category', 'City', 'Price', 'Rating']])
    
    # --- PERBAIKAN ERROR MAP DI SINI ---
    if not filtered_df.empty:
        # Kita buat dataframe baru khusus untuk map dengan nama kolom yang sesuai standar Streamlit (lat & lon)
        map_data = filtered_df[['Lat', 'Long']].rename(columns={'Lat': 'lat', 'Long': 'lon'})
        st.map(map_data)