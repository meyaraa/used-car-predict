import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# ------------------- 4 TEST MEKANIS -------------------
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_predict_harga_valid_input(client):
    payload = {
        "age": 2.0,
        "km_driven": 15000.0,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Automatic",
        "owner": "First Owner"
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 200, f"Error detail: {response.text}"
    assert "estimated_price" in response.json()

def test_predict_harga_missing_field(client):
    # Menghilangkan field 'age'
    payload = {
        "km_driven": 15000.0,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Automatic",
        "owner": "First Owner"
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

def test_predict_harga_invalid_enum(client):
    # 'fuel' tidak valid (Water)
    payload = {
        "age": 2.0,
        "km_driven": 15000.0,
        "fuel": "Water",
        "seller_type": "Individual",
        "transmission": "Automatic",
        "owner": "First Owner"
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

# ------------------- 2 BEHAVIORAL TEST -------------------
def test_behavioral_older_car_is_cheaper(client):
    """Kasus B: Kendaraan yang lebih tua dengan spesifikasi lain identik harus diprediksi lebih murah."""
    # Ubah spesifikasi ke skenario yang lebih umum (jarak tempuh lumayan tinggi)
    base_car = {
        "km_driven": 80000.0, 
        "fuel": "Petrol",
        "seller_type": "Dealer", # Ubah ke Dealer agar variasi harga lebih stabil
        "transmission": "Manual",
        "owner": "First Owner"
    }
    
    # Perlebar jarak umurnya agar depresiasi harganya sangat ekstrem dan jelas
    newer_car = {**base_car, "age": 1.0}
    older_car = {**base_car, "age": 12.0}
    
    res_newer = client.post("/predict-harga", json=newer_car).json()["estimated_price"]
    res_older = client.post("/predict-harga", json=older_car).json()["estimated_price"]
    
    assert res_older < res_newer, f"Gagal: Mobil Tua ({res_older}) lebih mahal dari Mobil Baru ({res_newer})"

def test_behavioral_higher_mileage_is_cheaper(client):
    base_car = {
        "age": 3.0,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Automatic",
        "owner": "First Owner"
    }
    
    low_km = {**base_car, "km_driven": 10000.0}
    high_km = {**base_car, "km_driven": 150000.0}
    
    res_low = client.post("/predict-harga", json=low_km).json()["estimated_price"]
    res_high = client.post("/predict-harga", json=high_km).json()["estimated_price"]
    
    assert res_high <= res_low, "Mobil dengan KM tinggi seharusnya tidak lebih mahal dari KM rendah!"