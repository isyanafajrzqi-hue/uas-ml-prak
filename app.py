import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# LOAD MODEL & SCALER
# ============================================
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('best_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="wide")
st.title("❤️ Prediksi Risiko Kardiovaskular")
st.markdown("---")

# Buat tab
tab1, tab2 = st.tabs(["🔍 Prediksi & Visualisasi", "📊 Hasil Training"])

# ============================================
# TAB 1: PREDIKSI + VISUALISASI INTERAKTIF
# ============================================
with tab1:
    st.subheader("Masukkan Data Pasien")
    
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Usia (29 - 65 tahun)", 
            min_value=29, max_value=65, value=50, step=1,
            help="Rentang data asli: 29 - 65 tahun"
        )
        gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
        height = st.number_input(
            "Tinggi (145 - 190 cm)", 
            min_value=145, max_value=190, value=170, step=1,
            help="Rentang data asli: 145 - 190 cm"
        )
        weight = st.number_input(
            "Berat (50 - 120 kg)", 
            min_value=50, max_value=120, value=70, step=1,
            help="Rentang data asli: 50 - 120 kg"
        )
        
    with col2:
        ap_hi = st.number_input(
            "Sistolik (100 - 180 mmHg)", 
            min_value=100, max_value=180, value=120, step=1,
            help="Rentang data asli: 100 - 180 mmHg"
        )
        ap_lo = st.number_input(
            "Diastolik (60 - 120 mmHg)", 
            min_value=60, max_value=120, value=80, step=1,
            help="Rentang data asli: 60 - 120 mmHg"
        )
        cholesterol = st.selectbox(
            "Kolesterol", 
            ["Normal", "Above Normal", "Well Above Normal"],
            help="1=Normal, 2=Above Normal, 3=Well Above Normal"
        )
        gluc = st.selectbox(
            "Gula Darah", 
            ["Normal", "Above Normal", "Well Above Normal"],
            help="1=Normal, 2=Above Normal, 3=Well Above Normal"
        )

    col3, col4 = st.columns(2)
    with col3:
        smoke = st.selectbox("Merokok", ["Tidak", "Ya"])
        alco = st.selectbox("Minum Alkohol", ["Tidak", "Ya"])
        active = st.selectbox("Aktivitas Fisik", ["Tidak Aktif", "Aktif"])

    # Mapping
    gender_map = {"Perempuan": 0, "Laki-laki": 1}
    chol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    smoke_map = {"Tidak": 0, "Ya": 1}
    alco_map = {"Tidak": 0, "Ya": 1}
    active_map = {"Tidak Aktif": 0, "Aktif": 1}

    if st.button("🔍 Prediksi Sekarang", type="primary"):
        # Buat input
        input_data = np.array([[
            age,
            gender_map[gender],
            height,
            weight,
            ap_hi,
            ap_lo,
            chol_map[cholesterol],
            gluc_map[gluc],
            smoke_map[smoke],
            alco_map[alco],
            active_map[active]
        ]])
        
        # Scaling
        input_scaled = scaler.transform(input_data)
        input_cnn = input_scaled.reshape(1, input_scaled.shape[1], 1)
        
        # Prediksi
        prediction = model.predict(input_cnn)
        prob = prediction[0][0]
        
        # ============================================
        # HASIL PREDIKSI
        # ============================================
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            if prob > 0.5:
                st.error(f"⚠️ **Risiko TINGGI**")
            else:
                st.success(f"✅ **Risiko RENDAH**")
            st.metric("Probabilitas Risiko", f"{prob*100:.2f}%")
            st.progress(float(prob))
        
        with col_res2:
            # Gauge chart sederhana pakai matplotlib
            fig_gauge, ax_gauge = plt.subplots(figsize=(4, 2))
            ax_gauge.barh(['Risiko'], [prob*100], color='red' if prob > 0.5 else 'green')
            ax_gauge.set_xlim(0, 100)
            ax_gauge.set_xlabel('Probabilitas Risiko (%)')
            ax_gauge.set_title('Tingkat Risiko')
            st.pyplot(fig_gauge)
            plt.close(fig_gauge)
        
        # ============================================
        # VISUALISASI INTERAKTIF (Berdasarkan Input User)
        # ============================================
        st.markdown("---")
        st.subheader("📈 Visualisasi Data Pasien")
        
        # Data untuk chart
        feature_names = ['Usia', 'Tinggi', 'Berat', 'Sistolik', 'Diastolik']
        user_values = [age, height, weight, ap_hi, ap_lo]
        
        # Nilai normal/rata-rata sebagai pembanding
        normal_values = [50, 170, 70, 120, 80]
        
        # Buat bar chart perbandingan
        fig_compare, ax_compare = plt.subplots(figsize=(10, 5))
        x = np.arange(len(feature_names))
        width = 0.35
        
        bars1 = ax_compare.bar(x - width/2, user_values, width, label='Nilai Pasien', color='#1f77b4')
        bars2 = ax_compare.bar(x + width/2, normal_values, width, label='Nilai Normal', color='#ff7f0e')
        
        ax_compare.set_xlabel('Fitur')
        ax_compare.set_ylabel('Nilai')
        ax_compare.set_title('Perbandingan Nilai Pasien vs Nilai Normal')
        ax_compare.set_xticks(x)
        ax_compare.set_xticklabels(feature_names)
        ax_compare.legend()
        ax_compare.grid(True, alpha=0.3)
        
        # Tambahkan nilai di atas bar
        for bar in bars1:
            height_val = bar.get_height()
            ax_compare.annotate(f'{height_val:.0f}', 
                               xy=(bar.get_x() + bar.get_width()/2, height_val),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        for bar in bars2:
            height_val = bar.get_height()
            ax_compare.annotate(f'{height_val:.0f}', 
                               xy=(bar.get_x() + bar.get_width()/2, height_val),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        st.pyplot(fig_compare)
        plt.close(fig_compare)
        
        # ============================================
        # KATEGORI RISIKO - Bar chart
        # ============================================
        fig_risk, ax_risk = plt.subplots(figsize=(8, 4))
        
        risk_levels = ['Rendah', 'Sedang', 'Tinggi']
        risk_values = [0, 0, 0]
        if prob < 0.3:
            risk_values[0] = 1
        elif prob < 0.7:
            risk_values[1] = 1
        else:
            risk_values[2] = 1
        
        colors_risk = ['green' if prob < 0.3 else 'lightgray',
                      'yellow' if 0.3 <= prob < 0.7 else 'lightgray',
                      'red' if prob >= 0.7 else 'lightgray']
        
        ax_risk.bar(risk_levels, risk_values, color=colors_risk)
        ax_risk.set_title('Kategori Risiko')
        ax_risk.set_ylim(0, 1.2)
        ax_risk.set_ylabel('Status')
        ax_risk.text(0, 1.05, 'Aman' if prob < 0.3 else '', ha='center')
        ax_risk.text(1, 1.05, 'Waspada' if 0.3 <= prob < 0.7 else '', ha='center')
        ax_risk.text(2, 1.05, 'Bahaya' if prob >= 0.7 else '', ha='center')
        
        st.pyplot(fig_risk)
        plt.close(fig_risk)
        
        # ============================================
        # FITUR DOWNLOAD
        # ============================================
        st.markdown("---")
        st.subheader("📥 Download Data Prediksi")
        
        download_data = {
            'Fitur': ['Usia (tahun)', 'Jenis Kelamin', 'Tinggi (cm)', 'Berat (kg)', 
                      'Sistolik (mmHg)', 'Diastolik (mmHg)', 'Kolesterol', 
                      'Gula Darah', 'Merokok', 'Minum Alkohol', 'Aktivitas Fisik'],
            'Nilai': [age, gender, height, weight, ap_hi, ap_lo, cholesterol, 
                      gluc, smoke, alco, active]
        }
        
        download_data['Fitur'].append('Probabilitas Risiko')
        download_data['Nilai'].append(f'{prob*100:.2f}%')
        download_data['Fitur'].append('Status')
        download_data['Nilai'].append('Risiko TINGGI' if prob > 0.5 else 'Risiko RENDAH')
        
        df_download = pd.DataFrame(download_data)
        csv_buffer = io.StringIO()
        df_download.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode('utf-8-sig')
        
        st.download_button(
            label="📥 Download Data Prediksi (CSV)",
            data=csv_data,
            file_name=f'prediksi_jantung_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            use_container_width=True
        )

# ============================================
# TAB 2: HASIL TRAINING (GAMBAR STATIS)
# ============================================
with tab2:
    st.header("📊 Visualisasi Hasil Training")
    
    image_files = {
        "training_plots.png": "Akurasi dan Loss",
        "confusion_matrix.png": "Confusion Matrix",
        "comparison_plot.png": "Perbandingan Performa"
    }
    
    for file, caption in image_files.items():
        if os.path.exists(file):
            st.image(file, caption=caption, use_container_width=True)
        else:
            st.warning(f"File {file} tidak ditemukan. Jalankan training terlebih dahulu.")
    
    # Tampilkan ringkasan model - DIPERBAIKI
    st.subheader("📄 Ringkasan Model")
    if os.path.exists('model_summary.txt'):
        try:
            with open('model_summary.txt', 'r', encoding='utf-8') as f:
                summary = f.read()
            st.code(summary, language='text')
        except:
            with open('model_summary.txt', 'r', encoding='latin-1') as f:
                summary = f.read()
            st.code(summary, language='text')
    else:
        st.warning("File model_summary.txt tidak ditemukan.")