export FLASK_ENV=production
export MODEL_PATH=models/model.joblib
export SCALER_PATH=models/scaler.joblib
export THRESHOLD=0.75
gunicorn -w 2 -b 0.0.0.0:${PORT:-5000} api.app:app --timeout 60