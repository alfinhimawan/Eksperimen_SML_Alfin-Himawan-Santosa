import pandas as pd
import os

def run_preprocessing():
    print("Memulai proses preprocessing data...")
    
    # Menentukan path file (menggunakan absolute/relative path yang aman)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(base_dir, '../dataset_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    clean_data_dir = os.path.join(base_dir, 'dataset_preprocessing')
    clean_data_path = os.path.join(clean_data_dir, 'dataset_clean.csv')
    
    # Buat folder output jika belum ada
    os.makedirs(clean_data_dir, exist_ok=True)
    
    # 1. Load raw dataset
    print("Membaca data raw...")
    df = pd.read_csv(raw_data_path)
    
    # 2. Data Cleaning
    print("Melakukan pembersihan data...")
    # Drop kolom customerID karena tidak relevan untuk machine learning
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Kolom TotalCharges memiliki spasi kosong (" ") yang harus diubah menjadi angka
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Isi nilai kosong yang dihasilkan dengan angka 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 3. Encoding Kategorikal
    print("Melakukan encoding pada kolom kategorikal...")
    # Ubah target 'Churn' menjadi 1 (Yes) dan 0 (No)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Lakukan One-Hot Encoding untuk sisa kolom kategorikal
    df = pd.get_dummies(df, drop_first=True)
    
    # 4. Simpan dataset yang sudah bersih
    print(f"Menyimpan data bersih ke: {clean_data_path}")
    df.to_csv(clean_data_path, index=False)
    
    print("Preprocessing selesai! Data siap dilatih.")

if __name__ == '__main__':
    run_preprocessing()
