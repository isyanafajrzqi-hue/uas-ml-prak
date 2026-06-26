import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load model dan scaler
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('best_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="centered")
st.title("❤️ Prediksi Risiko Kardiovaskular")
st.markdown("---")

# Input user
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia (tahun)", min_value=20, max_value=80, value=50)
    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    height = st.number_input("Tinggi (cm)", min_value=140, max_value=200, value=170)
    weight = st.number_input("Berat (kg)", min_value=40, max_value=150, value=70)
    
with col2:
    ap_hi = st.number_input("Sistolik (mmHg)", min_value=90, max_value=200, value=120)
    ap_lo = st.number_input("Diastolik (mmHg)", min_value=60, max_value=140, value=80)
    cholesterol = st.selectbox("Kolesterol", ["Normal", "Above Normal", "Well Above Normal"])
    gluc = st.selectbox("Gula Darah", ["Normal", "Above Normal", "Well Above Normal"])

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
    # Buat input sesuai urutan fitur
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
    
    # Reshape untuk CNN
    input_cnn = input_scaled.reshape(1, input_scaled.shape[1], 1)
    
    # Prediksi
    prediction = model.predict(input_cnn)
    prob = prediction[0][0]
    
    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")
    
    if prob > 0.5:
        st.error(f"⚠️ **Risiko TINGGI** (Probabilitas: {prob*100:.2f}%)")
        st.warning("Segera konsultasikan ke dokter untuk pemeriksaan lebih lanjut.")
    else:
        st.success(f"✅ **Risiko RENDAH** (Probabilitas: {prob*100:.2f}%)")
        st.info("Jaga pola makan dan rutin berolahraga.")
    
    # Progress bar
    st.progress(float(prob))