import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Used Car Price API (Truecars Data)",
    version="0.1.0"
)

# Load model saat aplikasi berjalan
try:
    model = joblib.load("models/model.joblib")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Schema Validasi Input
class CarInput(BaseModel):
    brand: str
    age: float = Field(..., ge=0, le=50)
    km: float = Field(..., ge=0)
    has_accident: int = Field(..., ge=0, le=1)
    is_first_owner: int = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "brand": "Toyota",
                    "age": 4.0,
                    "km": 50000.0,
                    "has_accident": 0,
                    "is_first_owner": 1
                }
            ]
        }
    }

# Endpoint 1: Info Layanan
@app.get("/")
def info_layanan():
    return {
        "nama_layanan": "API Prediksi Harga Mobil Bekas (Truecars)",
        "deskripsi": "Melayani estimasi harga mobil bekas menggunakan model Machine Learning berbasis Random Forest/Ridge.",
        "status": "Aktif",
        "dokumentasi_api": "Silakan akses /docs untuk mencoba API"
    }

# Endpoint 2: Health Check
@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model tidak ditemukan atau gagal dimuat.")
    return {"status": "healthy", "model_loaded": True}

# Endpoint 3: Prediksi Harga (Dual-Currency)
@app.post("/predict-harga")
def predict_harga(data: CarInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model belum siap.")
    
    # Ubah input menjadi DataFrame
    input_data = pd.DataFrame([data.model_dump()])
    
    # Lakukan Prediksi (Output asli dalam USD)
    pred_usd = float(model.predict(input_data)[0])
   
    if pred_usd < 500:
        pred_usd = 500.0
    
    # Konstanta RMSE dari hasil evaluasi test set 
    rmse_usd = 13723.73 
    
    # Konversi ke IDR 
    kurs_idr = 18000
    pred_idr = pred_usd * kurs_idr
    rmse_idr = rmse_usd * kurs_idr
    
    # Kembalikan response Dual-Currency
    return {
        "status": "success",
        "estimated_price_usd": round(pred_usd, 2),
        "estimated_price_idr": round(pred_idr, 2),
        "currency_rate_applied": kurs_idr,
        "confidence_margin_usd": round(rmse_usd, 2),
        "confidence_margin_idr": round(rmse_idr, 2),
        "message": f"Estimasi harga adalah ${pred_usd:,.2f} (sekitar Rp {pred_idr:,.0f})."
    }