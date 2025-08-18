import os, json
from glob import glob
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoModelForMaskedImageModeling
from multiprocessing import Process

# --- BEiT-3 setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("microsoft/beit-3-base")
beit = AutoModelForMaskedImageModeling.from_pretrained("microsoft/beit-3-base").to(device).eval()

# --- DeepSolo setup ---
# Sau khi clone/install theo repo chính thức
from deepSolo import build_deepsolo  # real import
deepsolo = build_deepsolo("ViTAEv2-S", pretrained=True).to(device).eval()

# --- PARSeq setup ---
# Sau khi clone/install theo repo chính thức
from parseq import Model as PARSeqModel
parseq = PARSeqModel(pretrained=True).to(device).eval()

# --- Dataset class ---
class FrameDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        return p, Image.open(p).convert("RGB")

# --- BEiT-3 embedding ---
def extract_beit3(img_paths, out_path, batch_size=16):
    ds = FrameDataset(img_paths)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=lambda x: list(zip(*x)))
    embs = []
    for _, imgs in tqdm(dl, desc=f"BEiT-3 {os.path.basename(out_path)}"):
        with torch.no_grad():
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            out = beit(**inputs).logits.mean(1).cpu().numpy()
        embs.append(out)
    np.save(out_path, np.vstack(embs))

# --- Scene Text Recognition Pipeline ---
def run_str(img_paths, out_json):
    results = {}
    for p in tqdm(img_paths, desc=f"STR {os.path.basename(out_json)}"):
        img = Image.open(p).convert("RGB")
        img_t = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            bboxes = deepsolo(img_t)  # real API
            texts = []
            for box in bboxes:
                x1,y1,x2,y2 = map(int, box)
                crop = img.crop((x1,y1,x2,y2))
                texts.append(parseq.recognize(crop))  # real API
        results[os.path.basename(p)] = texts
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# --- Process per video folder in parallel ---
def process_video(video_dir, out_feat, out_txt, batch_size=16):
    vid = os.path.basename(video_dir)
    img_paths = sorted(glob(os.path.join(video_dir, "*.jpg")))
    if not img_paths:
        return
    os.makedirs(out_feat, exist_ok=True)
    os.makedirs(out_txt, exist_ok=True)
    feat_f = os.path.join(out_feat, f"{vid}.npy")
    txt_f = os.path.join(out_txt, f"{vid}.json")

    p1 = Process(target=extract_beit3, args=(img_paths, feat_f, batch_size))
    p2 = Process(target=run_str, args=(img_paths, txt_f))
    p1.start(); p2.start()
    p1.join(); p2.join()

def main(root_keyframes, out_feat, out_txt):
    for root, dirs, files in os.walk(root_keyframes):
        if glob(os.path.join(root, "*.jpg")):
            print(f"Processing {root}")
            process_video(root, out_feat, out_txt)

if __name__ == "__main__":
    root_kf = "/mnt/data"  # nơi chứa Keyframes_L** folders
    out_f = "/mnt/data/beit_features"
    out_t = "/mnt/data/str_text"
    main(root_kf, out_f, out_t)
