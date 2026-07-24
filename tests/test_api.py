import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_harga_valid_input(client):
    payload = {"brand": "Toyota", "age": 4.0, "km": 30000.0, "has_accident": 0, "is_first_owner": 1}
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 200

def test_predict_harga_missing_field(client):
    payload = {"brand": "Toyota", "km": 30000.0, "has_accident": 0, "is_first_owner": 1} # 'age' dihapus
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

def test_predict_harga_invalid_range(client):
    payload = {"brand": "Toyota", "age": 4.0, "km": -100.0, "has_accident": 0, "is_first_owner": 1} # km negatif tidak logis
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

def test_behavioral_older_car_is_cheaper(client):
    base_car = {"brand": "Honda", "km": 40000.0, "has_accident": 0, "is_first_owner": 1}
    newer_car = {**base_car, "age": 2.0}
    older_car = {**base_car, "age": 10.0}
    
    res_newer = client.post("/predict-harga", json=newer_car).json()["estimated_price"]
    res_older = client.post("/predict-harga", json=older_car).json()["estimated_price"]
    assert res_older < res_newer

def test_behavioral_accident_car_is_cheaper(client):
    base_car = {"brand": "Chevrolet", "age": 5.0, "km": 50000.0, "is_first_owner": 1}
    clean_car = {**base_car, "has_accident": 0}
    crashed_car = {**base_car, "has_accident": 1}
    
    res_clean = client.post("/predict-harga", json=clean_car).json()["estimated_price"]
    res_crashed = client.post("/predict-harga", json=crashed_car).json()["estimated_price"]
    assert res_crashed < res_clean, "Mobil tabrakan seharusnya lebih murah!"

def test_info_layanan_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "nama_layanan" in response.json()