import os
import io
import time
import joblib
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template
from facenet_pytorch import InceptionResnetV1, MTCNN
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.joblib')
SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.joblib')
THRESHOLD = float(os.getenv('THRESHOLD', '0.75'))
MODEL_VERSION = os.getenv('MODEL_VERSION', 'me-verifier-v1')
MAX_MB = float(os.getenv('MAX_MB', '5'))
ALLOWED = {'image/jpeg', 'image/png'}

app = Flask(__name__, template_folder='template')

# Load detector, model, and classifier
mtcnn = MTCNN(keep_all=True, device='cpu')
inception = InceptionResnetV1(pretrained='vggface2').eval()
clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def read_image(file_storage):
    data = file_storage.read()
    return Image.open(io.BytesIO(data)).convert('RGB')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok', 'model_version': MODEL_VERSION})

def process_image(file):
    """Procesa una imagen y devuelve el dict de resultado o error."""
    start = time.time()
    if file.mimetype not in ALLOWED:
        return {'error': 'solo image/jpeg o image/png'}, 400

    # size check
    file.seek(0, 2)
    size = file.tell() / (1024 * 1024)
    if size > MAX_MB:
        return {'error': f'archivo demasiado grande > {MAX_MB} MB'}, 400
    file.seek(0)
    try:
        img = read_image(file)
    except Exception:
        return {'error': 'imagen no legible'}, 400

    boxes, probs = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return {'error': 'no_face_detected'}, 400

    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    idx = int(max(range(len(areas)), key=lambda i: areas[i]))
    box = boxes[idx].astype(int)
    crop = img.crop((box[0], box[1], box[2], box[3])).resize((160,160))
    arr = np.asarray(crop).transpose((2,0,1)) / 255.0
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = inception(tensor).cpu().numpy()
    emb_s = scaler.transform(emb)
    score = float(clf.predict_proba(emb_s)[0,1])
    is_me = bool(score >= THRESHOLD)
    timing_ms = (time.time() - start) * 1000

    return {
        'model_version': MODEL_VERSION,
        'is_me': is_me,
        'score': round(score, 3),
        'threshold': THRESHOLD,
        'timing_ms': round(timing_ms, 3)
    }, 200


@app.route('/verify', methods=['POST'])
def verify():
    start = time.time()
    if 'image' not in request.files:
        return jsonify({'error': 'missing file param "image"'}), 400
    f = request.files['image']
    if f.mimetype not in ALLOWED:
        return jsonify({'error': 'solo image/jpeg o image/png'}), 400

    f.seek(0, 2)
    size = f.tell() / (1024 * 1024)
    if size > MAX_MB:
        return jsonify({'error': f'archivo demasiado grande > {MAX_MB} MB'}), 400
    f.seek(0)

    try:
        img = read_image(f)
    except Exception:
        return jsonify({'error': 'imagen no legible'}), 400

    boxes, probs = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return jsonify({'error': 'no_face_detected'}), 400

    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    idx = int(max(range(len(areas)), key=lambda i: areas[i]))
    box = boxes[idx].astype(int)
    crop = img.crop((box[0], box[1], box[2], box[3])).resize((160, 160))

    arr = np.asarray(crop).transpose((2, 0, 1)) / 255.0
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = inception(tensor).cpu().numpy()

    emb_s = scaler.transform(emb)
    score = float(clf.predict_proba(emb_s)[0, 1])
    is_me = bool(score >= THRESHOLD)

    timing_ms = (time.time() - start) * 1000
    resp = {
        'model_version': MODEL_VERSION,
        'is_me': is_me,
        'score': round(score, 3),
        'threshold': THRESHOLD,
        'timing_ms': round(timing_ms, 3)
    }
    return jsonify(resp), 200


@app.route('/verify_ui', methods=['POST'])
def verify_ui():
    if 'image' not in request.files:
        return render_template('index.html', error="Debe subir una imagen.")

    result, status = process_image(request.files['image'])

    # Si hubo error en la verificación
    if 'error' in result:
        return render_template('index.html', error=result['error'])

    # Si la verificación fue exitosa
    return render_template('index.html', 
                           result=result,
                           is_me=result['is_me'],
                           score=result['score'],
                           threshold=result['threshold'],
                           timing=result['timing_ms'])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)