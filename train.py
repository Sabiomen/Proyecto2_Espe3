import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import json
import argparse
import os

def load_labels(csv_path):
    import csv
    X=[]
    y=[]
    with open(csv_path) as f:
        r=csv.reader(f)
        next(r)
        for path,label in r:
            y.append(int(label))
    return np.array(y)

def main(emb_path='data/embeddings.npy', labels_csv='data/labels.csv', model_out='models/model.joblib', scaler_out='models/scaler.joblib', metrics_out='reports/metrics.json'):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    X = np.load(emb_path)
    y = load_labels(labels_csv)

    # train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_val_s)
    y_proba = clf.predict_proba(X_val_s)[:,1]

    metrics = {
        'accuracy': float(accuracy_score(y_val, y_pred)),
        'roc_auc': float(roc_auc_score(y_val, y_proba)),
        'n_train': int(len(y_train)),
        'n_val': int(len(y_val))
    }

    joblib.dump(clf, model_out)
    joblib.dump(scaler, scaler_out)
    with open(metrics_out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved model and metrics:", metrics)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', default='data/embeddings.npy')
    parser.add_argument('--labels', default='data/labels.csv')
    parser.add_argument('--model_out', default='models/model.joblib')
    parser.add_argument('--scaler_out', default='models/scaler.joblib')
    parser.add_argument('--metrics_out', default='reports/metrics.json')
    args = parser.parse_args()
    main(args.embeddings, args.labels, args.model_out, args.scaler_out, args.metrics_out)