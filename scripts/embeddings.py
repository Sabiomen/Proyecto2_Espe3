import os
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import csv
from tqdm import tqdm
import torch

def image_to_embedding(model, img_path):
    img = Image.open(img_path).convert('RGB').resize((160,160))
    arr = np.asarray(img).transpose((2,0,1)) / 255.0
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    # normalize to model's expected range:
    embedding = model(tensor).detach().cpu().numpy()[0]
    return embedding

def main(root_dir, out_embeddings, out_csv):
    model = InceptionResnetV1(pretrained='vggface2').eval()
    embeddings = []
    rows = []
    for label_folder in ['me', 'not_me']:
        folder = Path(root_dir) / label_folder
        if not folder.exists():
            continue
        for p in tqdm(list(folder.glob('*'))):
            try:
                emb = image_to_embedding(model, p)
            except Exception as e:
                print("skip", p, e)
                continue
            embeddings.append(emb)
            rows.append([str(p), 1 if label_folder=='me' else 0])
    X = np.stack(embeddings)
    np.save(out_embeddings, X)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path','label'])
        writer.writerows(rows)
    print("Saved", out_embeddings, out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/cropped')
    parser.add_argument('--out_embeddings', default='data/embeddings.npy')
    parser.add_argument('--out_csv', default='data/labels.csv')
    args = parser.parse_args()
    main(args.root, args.out_embeddings, args.out_csv)