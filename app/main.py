import os
import json
import joblib
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "models/model.joblib"
META_PATH = "models/metadata.json"
model_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        model_data["pipeline"] = joblib.load(MODEL_PATH)
        with open(META_PATH, "r") as f:
            model_data["meta"] = json.load(f)
    yield
    model_data.clear()

app = FastAPI(title="Used Car Price API (Truecars Data)", lifespan=lifespan)

class CarPredictionInput(BaseModel):
    brand: str = Field(..., description="Merek mobil (contoh: Toyota, Honda, Kia, Chevrolet)")
    age: float = Field(..., ge=0, le=50, description="Umur mobil dalam hitungan tahun")
    km: float = Field(..., ge=0, description="Jarak tempuh dalam satuan kilometer")
    has_accident: int = Field(..., ge=0, le=1, description="1 jika pernah tabrakan/kecelakaan, 0 jika bersih")
    is_first_owner: int = Field(..., ge=0, le=1, description="1 jika kepemilikan tangan pertama, 0 jika bukan")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "brand": "Toyota",
                "age": 4.0,
                "km": 50000.0,
                "has_accident": 0,
                "is_first_owner": 1
            }
        }
    )
@app.get("/")
def info_layanan():
    return {
        "nama_layanan": "API Prediksi Harga Mobil Bekas (Truecars)",
        "deskripsi": "Melayani estimasi harga mobil bekas (dalam Rupiah) menggunakan model Machine Learning berbasis Random Forest/Ridge.",
        "status": "Aktif",
        "dokumentasi_api": "Silakan akses /docs untuk mencoba API"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy" if "pipeline" in model_data else "unhealthy", "model_loaded": "pipeline" in model_data}

@app.post("/predict-harga")
def predict_harga(payload: CarPredictionInput):
    if "pipeline" not in model_data:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    
    input_data = pd.DataFrame([payload.model_dump()])
    logger.info(f"Menerima request prediksi: {payload.model_dump()}")
    
    try:
        prediction = model_data["pipeline"].predict(input_data)[0]
        final_price_usd = max(0.0, float(prediction))
        rmse_margin_usd = model_data["meta"].get("test_rmse", 0)
        
        # --- KONVERSI MATA UANG USD KE IDR ---
        # Sumber kurs: Asumsi rata-rata nilai tukar USD ke IDR pada pertengahan 2026
        KURS_USD_KE_IDR = 18100.0 
        
        final_price_idr = final_price_usd * KURS_USD_KE_IDR
        rmse_margin_idr = rmse_margin_usd * KURS_USD_KE_IDR
        
        logger.info(f"Prediksi sukses: Rp {final_price_idr}")
        return {
            "status": "success",
            "estimated_price": round(final_price_idr, 2),
            "currency": "IDR",
            "confidence_margin_pm": round(rmse_margin_idr, 2),
            "message": f"Harga estimasi berada di rentang ± Rp {round(rmse_margin_idr, 2):,} berdasarkan RMSE test set."
        }
    except Exception as e:
        logger.error(f"Gagal melakukan prediksi: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))