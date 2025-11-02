import requests
import os

HOST = os.getenv('TEST_HOST','http://127.0.0.1:5000')

def test_health():
    r = requests.get(f'{HOST}/healthz')
    assert r.status_code == 200
    j = r.json()
    assert 'model_version' in j

def test_verify_no_file():
    r = requests.post(f'{HOST}/verify')
    assert r.status_code == 400

# For full tests, include sample images in tests/samples/ and run:
# r = requests.post(f'{HOST}/verify', files={'image': open('tests/samples/selfie.jpg','rb')})