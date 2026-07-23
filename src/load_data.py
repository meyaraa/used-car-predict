import os
import pandas as pd

DATA_PATH = "data/used_cars.csv"

def load_and_inspect_data():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File dataset tidak ditemukan di {DATA_PATH}. Unduh dataset Kaggle/UCI terlebih dahulu.")
        return None

    df = pd.read_csv(DATA_PATH)
    print("=== INFORMASI DATASET MENTAH ===")
    print(f"Jumlah Baris : {df.shape[0]}")
    print(f"Jumlah Kolom : {df.shape[1]}")
    print("\nTipe Data per Kolom:")
    print(df.dtypes)
    print("\nJumlah Nilai Hilang (Missing Values) per Kolom:")
    print(df.isnull().sum())
    
    return df

if __name__ == "__main__":
    load_and_inspect_data()