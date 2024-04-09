import torch, random
from tqdm import tqdm
import sys
import re
import os

gpu = 'cuda:0'
device = torch.device(gpu)
torch.cuda.empty_cache()


from transformers import CLIPProcessor, CLIPModel

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

import pandas as pd


root = '/notebooks/data/laion400m-meta/'
save_path = '/notebooks/data/laion400m-meta-embeds/'

class Collection(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, part):
        self.part = part
        filename = os.path.join(root, f'./{self.part:05d}.parquet')
        df = pd.read_parquet(filename)
        self.df = df[df["status"] == "success"]
        
        self.key_list = list(self.df["key"])
        self.caption_list = list(self.df["caption"])


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        key = self.key_list[idx]
        caption = self.caption_list[idx]

        return caption
    




if __name__ == "__main__":
    n_parts = 5

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for part in range(n_parts):
        cap = Collection(part)
        cap_dataloader = DataLoader(cap, batch_size=128, shuffle=False)
        text_representations = []
        for batch in tqdm(cap_dataloader):
        
            inputs = clip_processor(text=batch, return_tensors="pt", padding=True, truncation = True)
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = clip_model.get_text_features(**inputs)
            text_representation = outputs
            text_representation = text_representation / torch.norm(text_representation, dim = 1, keepdim = True)
            text_representations.append(text_representation)
        text_representations_tensor = torch.cat(text_representations, dim = 1)
        torch.save(text_representations_tensor, os.path.join(save_path, f'./{part:05d}_clip.pt'))