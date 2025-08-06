from fastapi.testclient import TestClient

from src.api.main import app  # Import your FastAPI app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Iris model API is running."}


def test_predict_endpoint():
    # This test will run but might fail if the model isn't loaded.
    # In a real CI/CD, you'd mock the model loading.
    # For this assignment, its presence is enough to demonstrate testing setup.
    payload = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert "class_name" in json_response
    assert json_response["prediction"] in [
        0,
        1,
        2,
    ]  # Should be one of the three classes
