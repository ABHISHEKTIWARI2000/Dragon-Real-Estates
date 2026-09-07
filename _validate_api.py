import sys

sys.path.insert(0, r"E:\ML projects\Dragon Real Estates")

from api.index import app

client = app.test_client()
response = client.post(
    "/api/predict",
    json={
        "features": [
            0.00632,
            18.0,
            2.31,
            0,
            0.538,
            6.575,
            65.2,
            4.09,
            1,
            296,
            15.3,
            396.9,
            4.98,
        ]
    },
)
print("status=", response.status_code)
print("json=", response.get_json())
