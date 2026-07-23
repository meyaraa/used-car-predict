import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from contextlib import asynccontextmanager

MODEL_PATH = "models/model.joblib"
model_data = {}

# Menggunakan lifespan sesuai standar FastAPI terbaru dan spesifikasi rubrik
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        model_data["pipeline"] = joblib.load(MODEL_PATH)
    else:
        model_data["pipeline"] = None
    yield
    model_data.clear()

app = FastAPI(
    title="Used Car Price Estimation API",
    description="API untuk memprediksi harga jual kendaraan bekas.",
    version="1.0.0",
    lifespan=lifespan
)

class CarPredictionInput(BaseModel):
    age: float = Field(..., ge=0, le=50, description="Umur kendaraan dalam tahun (0 - 50)")
    km_driven: float = Field(..., ge=0, description="Total kilometer tempuh")
    fuel: Literal["Petrol", "Diesel", "CNG", "LPG", "Electric"] = Field(..., description="Jenis bahan bakar")
    seller_type: Literal["Individual", "Dealer", "Trustmark Dealer"] = Field(..., description="Tipe penjual")
    transmission: Literal["Manual", "Automatic"] = Field(..., description="Tipe transmisi")
    owner: Literal["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"] = Field(..., description="Riwayat kepemilikan")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 3.0,
                "km_driven": 25000.0,
                "fuel": "Petrol",
                "seller_type": "Individual",
                "transmission": "Manual",
                "owner": "First Owner"
            }
        }
    )

@app.get("/")
def read_root():
    return {"message": "Selamat Datang di API Estimasi Harga Kendaraan Bekas"}

@app.get("/health")
def health_check():
    is_loaded = model_data.get("pipeline") is not None
    return {
        "status": "healthy" if is_loaded else "unhealthy",
        "model_loaded": is_loaded
    }

@app.post("/predict-harga")
def predict_harga(payload: CarPredictionInput):
    if model_data.get("pipeline") is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model belum siap/termuat."
        )
    
    input_data = pd.DataFrame([payload.model_dump()])
    
    try:
        prediction = model_data["pipeline"].predict(input_data)[0]
        
        # Tambahkan konversi dari Rupee ke Rupiah (1 INR = sekitar Rp 190)
        harga_rupiah = prediction * 190
        
        final_price = max(0.0, float(harga_rupiah))
        
        return {
            "status": "success",
            "estimated_price": round(final_price, 2),
            "currency": "IDR"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))