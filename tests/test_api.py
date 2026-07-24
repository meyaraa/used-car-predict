from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_info_layanan():
    response = client.get("/")
    assert response.status_code == 200
    assert "nama_layanan" in response.json()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] == True

def test_valid_input_dual_currency():
    payload = {"brand": "Toyota", "age": 4.0, "km": 50000.0, "has_accident": 0, "is_first_owner": 1}
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Pastikan API mengeluarkan dua mata uang
    assert "estimated_price_usd" in data
    assert "estimated_price_idr" in data
    assert data["currency_rate_applied"] == 18000

def test_missing_field():
    # Field 'age' sengaja dihapus
    payload = {"brand": "Toyota", "km": 50000.0, "has_accident": 0, "is_first_owner": 1} 
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

def test_invalid_range():
    # Field 'km' diisi negatif
    payload = {"brand": "Toyota", "age": 4.0, "km": -100.0, "has_accident": 0, "is_first_owner": 1} 
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

def test_behavioral_age():
    # Mobil identik, hanya beda umur
    payload_baru = {"brand": "Toyota", "age": 2.0, "km": 50000.0, "has_accident": 0, "is_first_owner": 1}
    payload_tua = {"brand": "Toyota", "age": 10.0, "km": 50000.0, "has_accident": 0, "is_first_owner": 1}
    
    res_baru = client.post("/predict-harga", json=payload_baru).json()["estimated_price_usd"]
    res_tua = client.post("/predict-harga", json=payload_tua).json()["estimated_price_usd"]
    
    assert res_tua < res_baru # Memastikan mobil tua harganya lebih murah

def test_behavioral_accident():
    # Mobil identik, hanya beda riwayat kecelakaan
    payload_bersih = {"brand": "Toyota", "age": 5.0, "km": 50000.0, "has_accident": 0, "is_first_owner": 1}
    payload_laka = {"brand": "Toyota", "age": 5.0, "km": 50000.0, "has_accident": 1, "is_first_owner": 1}
    
    res_bersih = client.post("/predict-harga", json=payload_bersih).json()["estimated_price_usd"]
    res_laka = client.post("/predict-harga", json=payload_laka).json()["estimated_price_usd"]
    
    assert res_laka < res_bersih # Memastikan mobil eks-kecelakaan harganya lebih murah