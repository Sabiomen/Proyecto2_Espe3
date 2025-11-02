import os
from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

def crop_folder(input_dir, output_dir, mtcnn):
    os.makedirs(output_dir, exist_ok=True)
    for img_path in tqdm(list(Path(input_dir).glob('*'))):
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"skip {img_path}: {e}")
            continue
        boxes, probs = mtcnn.detect(img)
        if boxes is None:
            continue
        # take the biggest box (largest area)
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        idx = int(max(range(len(areas)), key=lambda i: areas[i]))
        box = boxes[idx].astype(int)
        crop = img.crop((box[0], box[1], box[2], box[3])).resize((160,160))
        out_path = os.path.join(output_dir, img_path.name)
        crop.save(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    mtcnn = MTCNN(keep_all=True, device='cpu')
    crop_folder(args.input, args.output, mtcnn)

if __name__ == "__main__":
    main()