import pandas as pd
import os

def run_preprocessing():
    print("Memulai proses preprocessing data...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(base_dir, '../dataset_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    clean_data_dir = os.path.join(base_dir, 'dataset_preprocessing')
    clean_data_path = os.path.join(clean_data_dir, 'dataset_clean.csv')
    
    os.makedirs(clean_data_dir, exist_ok=True)
    
    print("Membaca data raw...")
    df = pd.read_csv(raw_data_path)
    
    print("Melakukan pembersihan data...")
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    print("Melakukan encoding pada kolom kategorikal...")
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)
    
    print(f"Menyimpan data bersih ke: {clean_data_path}")
    df.to_csv(clean_data_path, index=False)
    print("Preprocessing selesai! Data siap dilatih.")

if __name__ == '__main__':
    run_preprocessing()
