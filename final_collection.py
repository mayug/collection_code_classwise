import pandas as pd
import numpy as np
import argparse

import torch, random
from tqdm import tqdm
import sys
import re
import json
import os

from PIL import Image
import requests

LAION_LOCATION = "/shared/raiymbek/vlm_2/"

def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Running experiments"
    )
    parser.add_argument(
        "--b",
        dest="batch_num",
        help="batch_num",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--p",
        dest="part",
        help="part",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-cap",
        dest="cap",
        help="cap",
        action='store_true',
    )
    return parser.parse_args()

def get_limit(amounts):
    #return 0.3 * (amounts >= 500) + 0.3 * (amounts >= 1000) + 0.2 * (amounts >= 2000) + 0.2 * (amounts >= 5000)
    return 0.3 * (amounts >= 500) + 0.3 * (amounts >= 1000)
    
def run(part, batch_num):
    
    for batch_ind in tqdm(list(range(starting_ind, df_size, batch_num))):
        
        batch = df.iloc[batch_ind: min(batch_ind + batch_num, df_size)]
        sample_ids = batch["SAMPLE_ID"].tolist()
        scores = torch.load(f'{LAION_LOCATION}/laion400m_embeds/embeds/embeds_{part}/scores_{batch_ind}_{batch_num}.pt')
        if cap:
            class_done = (amounts >= 5000) + 0
            scores = scores * (1 - class_done) 
        maxes = torch.max(scores, dim = 1)
        max_val = maxes.values 
        max_ind = maxes.indices
    
        assignment = (max_ind + 1) * (max_val > get_limit(amounts)[max_ind]) - 1
        batch_assignments = torch.arange(assignment.shape[0])[assignment != -1]
        imagenet_assignments = assignment[assignment != -1]
        assignment_max = max_val[assignment != -1]
    
        for assign in range(len(batch_assignments)):
            batch_assign = batch_assignments[assign]
            imagenet_assign = imagenet_assignments[assign]
            assignment_max_val = float(assignment_max[assign].detach().numpy())
    
            class_to_assigments[imagenet_assign].append((sample_ids[batch_assign], assignment_max_val))
            amounts[imagenet_assign] += 1

        class_to_assigments[-1] = batch_ind + batch_num
        with open(f'results_final/{part}_assignment_{int(cap)}.json', 'w') as f:
            json.dump(class_to_assigments, f)        
    
if __name__ == "__main__":
    args = parse_args()
    
    part = args.part
    batch_num = args.batch_num

    cap = args.cap

    filename = f'{LAION_LOCATION}/laion400m-meta/part-{part:05d}-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet'
    df = pd.read_parquet(filename)
    df = df.dropna(subset=['TEXT'])
    df_size = df.shape[0]

    imagenet_classes = torch.load(f'class_embeddings/label_to_clip_class.pt').cpu()
    print("Loaded embeddings and parquet files")

    if f'{part}_assignment_{int(cap)}.json' in os.listdir("results_final/"):
        f = open(f'results_final/{part}_assignment_{int(cap)}.json')
        class_to_assigments = json.load(f)
    else:
        class_to_assigments = [[] for i in range(len(imagenet_classes)+1)]
        class_to_assigments[-1] = 0

    amounts = torch.tensor([0 for i in range(len(imagenet_classes))])
    for file in os.listdir("results_final/"):
        if f'_{int(cap)}.json' not in file:
            continue
            
        f = open(f'results_final/{file}')
        class_assigments_prev = json.load(f)[:-1]
        amounts += torch.tensor([len(np.unique(class_samples)) for class_samples in class_assigments_prev])
        
    print(torch.sum(amounts))
    
    starting_ind = class_to_assigments[-1]
    result = run(part, batch_num)
    