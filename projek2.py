import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Rekomendasi Wisata",
    page_icon="🏝️",
    layout="wide"
)

# Judul Aplikasi
st.title("🏝️ Sistem Rekomendasi Tempat Wisata Indonesia")
st.markdown("### User-Based Collaborative Filtering")
st.markdown("---")

# Data tempat wisata Indonesia (nama asli)
INDONESIA_TOURISM_PLACES = [
    "Pantai Kuta, Bali",
    "Pantai Sanur, Bali",
    "Pantai Nusa Dua, Bali",
    "Ubud, Bali",
    "Tanah Lot, Bali",
    "Uluwatu Temple, Bali",
    "Kintamani, Bali",
    "Candi Borobudur, Magelang",
    "Candi Prambanan, Yogyakarta",
    "Keraton Yogyakarta",
    "Malioboro, Yogyakarta",
    "Parangtritis Beach, Yogyakarta",
    "Bromo Tengger Semeru National Park",
    "Kawah Ijen, Banyuwangi",
    "Taman Nasional Baluran, Situbondo",
    "Danau Toba, Sumatera Utara",
    "Bukit Lawang, Sumatera Utara",
    "Pulau Samosir, Sumatera Utara",
    "Nias Island, Sumatera Utara",
    "Raja Ampat, Papua Barat",
    "Pulau Komodo, NTT",
    "Labuan Bajo, NTT",
    "Pulau Padar, NTT",
    "Pink Beach, Komodo",
    "Gili Trawangan, Lombok",
    "Gili Meno, Lombok",
    "Gili Air, Lombok",
    "Mount Rinjani, Lombok",
    "Senggigi Beach, Lombok",
    "Tanjung Tinggi Beach, Belitung",
    "Pulau Lengkuas, Belitung",
    "Danau Kelimutu, Flores",
    "Taman Nasional Bunaken, Sulawesi Utara",
    "Tana Toraja, Sulawesi Selatan",
    "Wakatobi National Park, Sulawesi Tenggara",
    "Derawan Islands, East Kalimantan",
    "Tanjung Puting National Park, Central Kalimantan",
    "Pramuka Island, Jakarta",
    "Tidung Island, Jakarta",
    "Taman Mini Indonesia Indah, Jakarta",
    "Ancol Dreamland, Jakarta",
    "Taman Safari Indonesia, Bogor",
    "Kebun Raya Bogor",
    "Kawah Putih, Bandung",
    "Tangkuban Perahu, Bandung",
    "Ciwidey Valley, Bandung",
    "Dago Dream Park, Bandung",
    "Braga Street, Bandung",
    "Taman Laut Banda, Maluku",
    "Fort Rotterdam, Makassar"
]

# Fungsi generate data dummy dengan nama tempat asli
def generate_tourism_data(num_users=100):
    """Generate dummy tourism rating data"""
    np.random.seed(42)
    
    data = []
    user_ids = list(range(1, num_users + 1))
    
    for user_id in user_ids:
        # Setiap user rating 15-30 tempat secara random
        num_ratings = np.random.randint(15, 31)
        rated_places = np.random.choice(
            INDONESIA_TOURISM_PLACES, 
            size=num_ratings, 
            replace=False
        )
        
        for place in rated_places:
            # Generate rating dengan distribusi normal
            base_rating = np.random.normal(3.5, 1.2)
            
            # Beberapa tempat populer dapat rating lebih tinggi
            popular_places = ["Pantai Kuta, Bali", "Candi Borobudur, Magelang", 
                            "Raja Ampat, Papua Barat", "Bromo Tengger Semeru National Park"]
            if place in popular_places:
                base_rating += np.random.uniform(0.5, 1.0)
            
            rating = int(np.clip(round(base_rating), 1, 5))
            
            data.append({
                'User_Id': user_id,
                'Place_Name': place,
                'Place_Ratings': rating
            })
    
    return pd.DataFrame(data)

# Inisialisasi session state
if 'all_ratings' not in st.session_state:
    st.session_state.all_ratings = generate_tourism_data()
if 'new_user_ratings' not in st.session_state:
    st.session_state.new_user_ratings = {}
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# Sidebar
st.sidebar.header("⚙️ Pengaturan Sistem")

# Main tabs
tab1, tab2, tab3 = st.tabs(["🎯 Dapatkan Rekomendasi", "⭐ Input Rating Baru", "📊 Data & Statistik"])

# TAB 1: DAPATKAN REKOMENDASI
with tab1:
    st.header("🎯 Dapatkan Rekomendasi Tempat Wisata")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input User")
        
        # Pilih metode
        method = st.radio(
            "Pilih user:",
            ["User baru (input rating dulu)", "User yang sudah ada"]
        )
        
        if method == "User baru (input rating dulu)":
            st.info("Pastikan sudah input rating di Tab 2 ⭐")
            
            if st.session_state.new_user_ratings:
                st.success(f"✅ Anda sudah memberi {len(st.session_state.new_user_ratings)} rating")
                st.write("Rating Anda:")
                for place, rating in list(st.session_state.new_user_ratings.items())[:5]:
                    stars = "⭐" * rating
                    st.write(f"- {place}: {stars}")
            else:
                st.warning("Belum ada rating. Silakan ke Tab 2 ⭐")
            
            user_type = "new"
            
        else:  # User yang sudah ada
            existing_users = st.session_state.all_ratings['User_Id'].unique()
            selected_user = st.selectbox(
                "Pilih ID User:",
                options=existing_users[:50]  # Batasi tampilan
            )
            
            # Tampilkan rating user terpilih
            user_ratings = st.session_state.all_ratings[
                st.session_state.all_ratings['User_Id'] == selected_user
            ]
            
            if not user_ratings.empty:
                st.write(f"**Rating User {selected_user}:**")
                for _, row in user_ratings.head(5).iterrows():
                    stars = "⭐" * row['Place_Ratings']
                    st.write(f"- {row['Place_Name']}: {stars}")
            
            user_type = "existing"
            selected_user_id = selected_user
        
        st.subheader("Parameter")
        num_recommendations = st.slider("Jumlah rekomendasi:", 5, 20, 10)
        similarity_threshold = st.slider("Threshold kemiripan:", 0.1, 0.9, 0.3, 0.1)
    
    with col2:
        st.subheader("Hasil Rekomendasi")
        
        if st.button("🚀 Generate Rekomendasi", type="primary", use_container_width=True):
            with st.spinner("Mencari rekomendasi terbaik untuk Anda..."):
                
                # Siapkan data
                if user_type == "new" and st.session_state.new_user_ratings:
                    # Gabungkan rating user baru dengan data existing
                    new_ratings_list = []
                    for place, rating in st.session_state.new_user_ratings.items():
                        new_ratings_list.append({
                            'User_Id': 9999,  # ID khusus untuk user baru
                            'Place_Name': place,
                            'Place_Ratings': rating
                        })
                    
                    temp_df = pd.concat([
                        st.session_state.all_ratings,
                        pd.DataFrame(new_ratings_list)
                    ], ignore_index=True)
                    
                    target_user = 9999
                    
                elif user_type == "existing":
                    temp_df = st.session_state.all_ratings
                    target_user = selected_user_id
                
                else:
                    st.error("Silakan beri rating terlebih dahulu di Tab 2 ⭐")
                    st.stop()
                
                # Buat user-item matrix
                user_item_matrix = temp_df.pivot_table(
                    index='User_Id',
                    columns='Place_Name',
                    values='Place_Ratings',
                    fill_value=0
                )
                
                # Hitung similarity
                user_similarity = cosine_similarity(user_item_matrix)
                user_similarity_df = pd.DataFrame(
                    user_similarity,
                    index=user_item_matrix.index,
                    columns=user_item_matrix.index
                )
                
                # Fungsi rekomendasi
                def get_recommendations(user_id, n=10):
                    if user_id not in user_similarity_df.index:
                        return None
                    
                    # Cari user similar
                    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
                    similar_users = similar_users[similar_users > similarity_threshold]
                    similar_users = similar_users.drop(user_id, errors='ignore')
                    
                    if len(similar_users) == 0:
                        return None
                    
                    # Tempat yang belum di-rating oleh user
                    user_rated = set(temp_df[temp_df['User_Id'] == user_id]['Place_Name'])
                    all_places = set(temp_df['Place_Name'])
                    unrated_places = list(all_places - user_rated)
                    
                    # Hitung prediksi rating
                    predictions = []
                    for place in unrated_places:
                        weighted_sum = 0
                        similarity_sum = 0
                        
                        for sim_user, similarity in similar_users.items():
                            user_rating = user_item_matrix.loc[sim_user, place]
                            if user_rating > 0:
                                weighted_sum += similarity * user_rating
                                similarity_sum += similarity
                        
                        if similarity_sum > 0:
                            predicted_rating = weighted_sum / similarity_sum
                            predictions.append({
                                'Tempat_Wisata': place,
                                'Prediksi_Rating': predicted_rating,
                                'Jumlah_User_Serupa': len(similar_users)
                            })
                    
                    # Urutkan dan ambil terbaik
                    predictions.sort(key=lambda x: x['Prediksi_Rating'], reverse=True)
                    return predictions[:n]
                
                # Dapatkan rekomendasi
                recommendations = get_recommendations(target_user, num_recommendations)
                
                if recommendations:
                    st.session_state.recommendations = recommendations
                    
                    # Tampilkan hasil dengan menarik
                    st.success(f"🎉 **{len(recommendations)} rekomendasi ditemukan!**")
                    
                    # Group berdasarkan prediksi rating
                    excellent = [r for r in recommendations if r['Prediksi_Rating'] >= 4.0]
                    good = [r for r in recommendations if 3.0 <= r['Prediksi_Rating'] < 4.0]
                    
                    if excellent:
                        st.subheader("🏆 **Rekomendasi Terbaik** (Rating ≥ 4.0)")
                        for i, rec in enumerate(excellent, 1):
                            with st.expander(f"{i}. {rec['Tempat_Wisata']}", expanded=(i==1)):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Prediksi Rating", f"{rec['Prediksi_Rating']:.2f}")
                                    # Visual stars
                                    stars_count = int(rec['Prediksi_Rating'])
                                    stars = "⭐" * stars_count
                                    if rec['Prediksi_Rating'] - stars_count >= 0.5:
                                        stars += "½"
                                    st.write(stars)
                                with col_b:
                                    st.metric("User Serupa", rec['Jumlah_User_Serupa'])
                                
                                # Tombol aksi
                                if st.button(f"💾 Simpan ke Wishlist", key=f"save_{i}"):
                                    st.success("Ditambahkan ke wishlist!")
                    
                    if good:
                        st.subheader("👍 **Rekomendasi Baik** (Rating 3.0-4.0)")
                        for i, rec in enumerate(good, 1):
                            st.write(f"**{i}. {rec['Tempat_Wisata']}**")
                            st.write(f"Prediksi: {rec['Prediksi_Rating']:.2f} | User serupa: {rec['Jumlah_User_Serupa']}")
                            st.progress(rec['Prediksi_Rating'] / 5)
                    
                    # Chart visualisasi
                    st.subheader("📊 Distribusi Prediksi Rating")
                    rec_df = pd.DataFrame(recommendations)
                    st.bar_chart(rec_df.set_index('Tempat_Wisata')['Prediksi_Rating'])
                    
                else:
                    st.error("""
                    **Tidak dapat menemukan rekomendasi yang cocok.**
                    
                    **Saran:**
                    1. Beri lebih banyak rating di Tab 2 ⭐
                    2. Turunkan threshold kemiripan
                    3. Coba user lain yang sudah ada
                    """)

# TAB 2: INPUT RATING BARU
with tab2:
    st.header("⭐ Input Rating Tempat Wisata Baru")
    st.markdown("Beri rating pada tempat wisata yang pernah Anda kunjungi:")
    
    # Reset button
    if st.session_state.new_user_ratings:
        if st.button("🔄 Reset Semua Rating"):
            st.session_state.new_user_ratings = {}
            st.rerun()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input rating untuk tempat-tempat tertentu
        st.subheader("Pilih Tempat Wisata")
        
        # Pencarian
        search_query = st.text_input("🔍 Cari tempat wisata:", "")
        
        # Filter places based on search
        if search_query:
            filtered_places = [p for p in INDONESIA_TOURISM_PLACES 
                             if search_query.lower() in p.lower()]
        else:
            filtered_places = INDONESIA_TOURISM_PLACES[:20]  # Tampilkan 20 pertama
        
        # Input rating untuk beberapa tempat
        selected_places = st.multiselect(
            "Pilih tempat yang pernah dikunjungi:",
            options=filtered_places,
            help="Pilih tempat wisata yang ingin Anda beri rating"
        )
        
        if selected_places:
            st.subheader("Beri Rating")
            
            for place in selected_places:
                # Skip jika sudah di-rating
                if place in st.session_state.new_user_ratings:
                    current_rating = st.session_state.new_user_ratings[place]
                    st.write(f"**{place}** - Rating saat ini: {'⭐' * current_rating}")
                    continue
                
                # Input rating baru
                cols = st.columns([3, 1])
                with cols[0]:
                    st.write(f"**{place}**")
                with cols[1]:
                    rating = st.select_slider(
                        "Rating:",
                        options=[1, 2, 3, 4, 5],
                        value=3,
                        key=f"rate_{place}"
                    )
                
                # Tombol simpan untuk tempat ini
                if st.button(f"💾 Simpan rating untuk {place[:30]}...", key=f"save_{place}"):
                    st.session_state.new_user_ratings[place] = rating
                    st.success(f"Rating {rating}⭐ untuk {place} disimpan!")
                    st.rerun()
        
        # Quick rating - rating cepat untuk tempat populer
        st.subheader("🎯 Rating Cepat (Tempat Populer)")
        
        popular_places = [
            "Pantai Kuta, Bali",
            "Candi Borobudur, Magelang", 
            "Bromo Tengger Semeru National Park",
            "Raja Ampat, Papua Barat",
            "Danau Toba, Sumatera Utara"
        ]
        
        for place in popular_places:
            if place not in st.session_state.new_user_ratings:
                cols = st.columns([4, 1])
                with cols[0]:
                    st.write(place)
                with cols[1]:
                    if st.button(f"⭐ Rate", key=f"quick_{place}"):
                        st.session_state.new_user_ratings[place] = 3  # Default 3 stars
                        st.rerun()
    
    with col2:
        st.subheader("📊 Rating Anda")
        
        if st.session_state.new_user_ratings:
            # Ringkasan rating
            total_places = len(st.session_state.new_user_ratings)
            avg_rating = sum(st.session_state.new_user_ratings.values()) / total_places
            
            st.metric("Total Tempat", total_places)
            st.metric("Rata-rata Rating", f"{avg_rating:.2f}")
            
            # Distribusi rating
            rating_counts = {1:0, 2:0, 3:0, 4:0, 5:0}
            for rating in st.session_state.new_user_ratings.values():
                rating_counts[rating] += 1
            
            st.write("**Distribusi:**")
            for stars in range(5, 0, -1):
                count = rating_counts[stars]
                if count > 0:
                    st.write(f"{'⭐' * stars}: {count} tempat")
            
            # Tempat dengan rating tertinggi
            if st.session_state.new_user_ratings:
                highest_rated = max(st.session_state.new_user_ratings.items(), 
                                  key=lambda x: x[1])
                st.write(f"**Favorit Anda:**")
                st.write(f"{highest_rated[0]}")
                st.write(f"{'⭐' * highest_rated[1]}")
        else:
            st.info("Belum ada rating")
            
            # Demo - generate random ratings
            if st.button("🎲 Coba Demo (Random Rating)"):
                demo_places = np.random.choice(INDONESIA_TOURISM_PLACES, 10, replace=False)
                for place in demo_places:
                    st.session_state.new_user_ratings[place] = np.random.randint(1, 6)
                st.success("Demo rating berhasil dibuat!")
                st.rerun()

# TAB 3: DATA & STATISTIK
with tab3:
    st.header("📊 Data & Statistik Tempat Wisata")
    
    # Statistik umum
    col1, col2, col3 = st.columns(3)
    with col1:
        total_users = st.session_state.all_ratings['User_Id'].nunique()
        st.metric("Total User Database", total_users)
    with col2:
        total_places = st.session_state.all_ratings['Place_Name'].nunique()
        st.metric("Total Tempat Wisata", total_places)
    with col3:
        avg_all_rating = st.session_state.all_ratings['Place_Ratings'].mean()
        st.metric("Rating Rata-rata", f"{avg_all_rating:.2f}")
    
    st.markdown("---")
    
    # Top tempat wisata
    st.subheader("🏆 Top 10 Tempat Wisata Terpopuler")
    
    # Hitung statistik per tempat
    place_stats = st.session_state.all_ratings.groupby('Place_Name').agg({
        'Place_Ratings': ['mean', 'count'],
        'User_Id': 'nunique'
    }).round(2)
    
    place_stats.columns = ['Rating_Rata', 'Jumlah_Rating', 'Jumlah_User']
    place_stats = place_stats.reset_index()
    
    # Filter untuk tempat dengan minimal 5 rating
    place_stats = place_stats[place_stats['Jumlah_Rating'] >= 5]
    
    # Tampilkan dalam dua kategori
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Berdasarkan Rating Tertinggi:**")
        top_by_rating = place_stats.sort_values('Rating_Rata', ascending=False).head(10)
        
        for i, (_, row) in enumerate(top_by_rating.iterrows(), 1):
            stars = "⭐" * int(row['Rating_Rata'])
            st.write(f"{i}. **{row['Place_Name']}**")
            cols = st.columns([3, 1])
            with cols[0]:
                st.write(f"   {stars} ({row['Rating_Rata']:.1f})")
            with cols[1]:
                st.write(f"👥 {int(row['Jumlah_User'])}")
    
    with col2:
        st.write("**Berdasarkan Popularitas:**")
        top_by_users = place_stats.sort_values('Jumlah_User', ascending=False).head(10)
        
        for i, (_, row) in enumerate(top_by_users.iterrows(), 1):
            stars = "⭐" * int(row['Rating_Rata'])
            st.write(f"{i}. **{row['Place_Name']}**")
            cols = st.columns([1, 2])
            with cols[0]:
                st.write(f"👥 {int(row['Jumlah_User'])}")
            with cols[1]:
                st.write(f"{stars} ({row['Rating_Rata']:.1f})")
    
    # Visualisasi
    st.markdown("---")
    st.subheader("📈 Visualisasi Data")
    
    # Grafik distribusi rating
    rating_dist = st.session_state.all_ratings['Place_Ratings'].value_counts().sort_index()
    st.write("**Distribusi Seluruh Rating:**")
    st.bar_chart(rating_dist)
    
    # Scatter plot rating vs jumlah user
    st.write("**Hubungan Rating vs Popularitas:**")
    st.scatter_chart(place_stats.set_index('Place_Name')[['Rating_Rata', 'Jumlah_User']])
    
    # Tampilkan data lengkap
    if st.checkbox("Tampilkan semua data statistik"):
        st.dataframe(place_stats.sort_values('Rating_Rata', ascending=False))

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🏝️ <b>Sistem Rekomendasi Tempat Wisata Indonesia</b> | User-Based Collaborative Filtering</p>
        <p><small>Input rating → Cari user serupa → Dapatkan rekomendasi!</small></p>
    </div>
    """,
    unsafe_allow_html=True
)

# CSS tambahan
st.markdown("""
<style>
    .stButton > button {
        border-radius: 10px;
        height: 3em;
    }
    .stSelectbox, .stMultiselect {
        margin-bottom: 1rem;
    }
    .st-expander {
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)