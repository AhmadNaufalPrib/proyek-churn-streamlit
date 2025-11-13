import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------
# 1. MEMUAT MODEL YANG SUDAH DILATIH
# ----------------------------------------------------
# Muat model pipeline dari file .pkl
# File ini harus ada di folder yang sama dengan app.py
try:
    model = joblib.load('model_churn_pipeline.pkl')
except FileNotFoundError:
    st.error("File 'model_churn_pipeline.pkl' tidak ditemukan. Pastikan ada di folder yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# ----------------------------------------------------
# 2. MEMBUAT JUDUL DAN FORM APLIKASI WEB
# ----------------------------------------------------
st.title('Aplikasi Prediksi Churn Pelanggan Telco')
st.write('Masukkan data pelanggan untuk memprediksi probabilitas churn.')

# Membuat form input di sidebar
st.sidebar.header('Input Data Pelanggan:')

# Kita akan membuat input untuk beberapa fitur paling penting
# Nama variabel (misal: 'tenure') HARUS SAMA PERSIS dengan nama kolom di data training

# --- Input Fitus ---
tenure = st.sidebar.slider('Lama Berlangganan (Bulan)', 0, 72, 12)
monthly_charges_rp = st.sidebar.slider(
    'Tagihan Bulanan (Rp)', 
    min_value=100000, 
    max_value=3000000,  # Asumsi max $200
    value=1000000,      # Nilai default
    step=50000          # Kelipatan
)
Contract = st.sidebar.selectbox('Tipe Kontrak', ['Month-to-month', 'One year', 'Two year'])
InternetService = st.sidebar.selectbox('Layanan Internet', ['DSL', 'Fiber optic', 'No'])
PaymentMethod = st.sidebar.selectbox('Metode Pembayaran', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Fitur lain yang dibutuhkan oleh model (kita isi dengan nilai default)
# Ini PENTING agar model tidak error. Kita harus memberi input SEMUA fitur.
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
Partner = st.sidebar.selectbox('Punya Pasangan?', ['Yes', 'No'])
Dependents = st.sidebar.selectbox('Punya Tanggungan?', ['Yes', 'No'])
TechSupport = st.sidebar.selectbox('Dukungan Teknis', ['Yes', 'No', 'No internet service'])
OnlineSecurity = st.sidebar.selectbox('Keamanan Online', ['Yes', 'No', 'No internet service'])
OnlineBackup = st.sidebar.selectbox('Backup Online', ['Yes', 'No', 'No internet service'])
DeviceProtection = st.sidebar.selectbox('Perlindungan Perangkat', ['Yes', 'No', 'No internet service'])
StreamingTV = st.sidebar.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
StreamingMovies = st.sidebar.selectbox('Streaming Film', ['Yes', 'No', 'No internet service'])
PhoneService = 'Yes' # Asumsi default
MultipleLines = 'No' # Asumsi default
PaperlessBilling = 'Yes' # Asumsi default
SeniorCitizen = 0 # Asumsi default

# TotalCharges akan di-handle oleh pipeline (atau diabaikan jika tidak dilatih)
# Untuk amannya, kita bisa buat estimasi kasar
 


# ----------------------------------------------------
# 3. MEMBUAT TOMBOL PREDIKSI
# ----------------------------------------------------
if st.sidebar.button('Prediksi Probabilitas Churn'):
    
    # --- TAMBAHKAN LOGIKA KONVERSI INI ---
    # Tentukan kurs konversi (misal $1 = Rp 15.000)
    KURS_USD_TO_IDR = 15000.0
    
    # Terjemahkan input Rupiah dari slider kembali ke Dolar
    # Ini adalah nilai yang akan dipahami oleh model
    monthly_charges_dollar = monthly_charges_rp / KURS_USD_TO_IDR
    
    # Hitung TotalCharges menggunakan nilai Dolar (karena model juga dilatih pakai ini)
    total_charges_dollar = monthly_charges_dollar * tenure
    # --- SELESAI MENAMBAHKAN ---


    # 3a. Kumpulkan semua data input ke dalam format DataFrame
    # Pastikan Anda mengganti nilai 'MonthlyCharges' dan 'TotalCharges'
    data_input = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': monthly_charges_dollar, # <-- PENTING: Gunakan nilai Dolar
        'TotalCharges': total_charges_dollar     # <-- PENTING: Gunakan nilai Dolar
    }
    
    # Ubah dictionary menjadi 1 baris DataFrame
    input_df = pd.DataFrame([data_input])

    # 3b. Lakukan prediksi
    try:
        # Gunakan model untuk memprediksi probabilitas
        # [0][1] berarti kita ambil probabilitas untuk kelas '1' (Churn)
        prob_churn = model.predict_proba(input_df)[0][1]
        persen_churn = prob_churn * 100

        # 3c. Tampilkan hasil
        st.subheader('Hasil Prediksi:')
        if persen_churn > 50:
            st.error(f'Probabilitas Churn: {persen_churn:.2f}% (Risiko Tinggi)')
            st.warning('Rekomendasi: Hubungi pelanggan ini untuk penawaran retensi.')
        else:
            st.success(f'Probabilitas Churn: {persen_churn:.2f}% (Risiko Rendah)')
            st.info('Pelanggan ini kemungkinan besar aman.')

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        st.write("Pastikan semua input diisi dengan benar.")