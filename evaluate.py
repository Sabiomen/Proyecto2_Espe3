import numpy as np
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
import matplotlib.pyplot as plt
import json
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

def main(emb_path='data/embeddings.npy', labels_csv='data/labels.csv', model_path='models/model.joblib', scaler_path='models/scaler.joblib', out_dir='reports'):
    os.makedirs(out_dir, exist_ok=True)
    X = np.load(emb_path)
    y = load_labels(labels_csv)

    scaler = joblib.load(scaler_path)
    clf = joblib.load(model_path)

    Xs = scaler.transform(X)
    y_proba = clf.predict_proba(Xs)[:,1]

    fpr, tpr, roc_th = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, pr_th = precision_recall_curve(y, y_proba)

    # plot ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f'ROC AUC = {roc_auc:.3f}')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.savefig(os.path.join(out_dir,'roc.png'))

    # PR
    plt.figure()
    plt.plot(recall, precision)
    plt.title('Precision-Recall')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.savefig(os.path.join(out_dir,'pr.png'))

    # choose threshold by maximizing F1
    from sklearn.metrics import f1_score
    best_t = 0.5
    best_f1 = 0
    for th in pr_th:
        preds = (y_proba >= th).astype(int)
        f1 = f1_score(y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = th

    cm = confusion_matrix(y, (y_proba>=best_t).astype(int)).tolist()
    metrics = {'roc_auc': float(roc_auc), 'best_threshold': float(best_t), 'best_f1': float(best_f1), 'confusion_matrix': cm}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved evaluation to", out_dir)
if __name__ == "__main__":
    main()