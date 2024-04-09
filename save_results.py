import torch, random
from tqdm import tqdm
import sys
import re
import os
import json

LAION_LOCATION = "/shared/raiymbek/vlm_2/"
PARTS = 12

def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Running experiments"
    )
    parser.add_argument(
        "--p",
        dest="part",
        help="how many parts to do",
        default=0,
        type=int,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    part_to_do = args.part

    all_assigments = []
    all_df = {}
    sample_to_part = {}
    
    print("Loading part files and collection results")
    for part in tqdm(range(part_to_do)):
            
        f = open(f'results_final/{part}_assignment_1.json')
        class_assigments = json.load(f)[:-1]
    
        filename = f'{LAION_LOCATION}/laion400m-meta/part-{part:05d}-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet'
        df = pd.read_parquet(filename)
        df = df.dropna(subset=['TEXT'])
        df_size = df.shape[0]
    
        for cl in class_assigments:
            for i in cl:
                sample_to_part[i[0]] = part
                part_to_sample[part].append(i[0])
    
        all_df[part] = df
        all_assigments.append(class_assigments)
    
    final_assignments = [[] for i in range(1000)]
    for assignment in all_assigments:
        for i in range(1000):
            final_assignments[i] += assignment[i]
    
    picked_assignments = [sorted(i, key = lambda x: -x[1])[:2000] for i in final_assignments]
    picked_assignments = [[j for j in i if not pd.isna(j[0])][:1000] for i in picked_assignments]
    
    part_to_sample = [[] for i in range(part_to_do)]
    sample_to_class = {}
    sample_to_sim = {}
    for cl_i in range(1000):
        cl = picked_assignments[cl_i]
        for i in cl:
            part = sample_to_part[i[0]]
            part_to_sample[part].append(i[0])
            sample_to_class[i[0]] = cl_i
            sample_to_sim[i[0]] = i[1]
    
    table = []
    for part in tqdm(range(PARTS)):
        sample_df = all_df[part]
        part_samples = part_to_sample[part]
        picked_df = sample_df[sample_df["SAMPLE_ID"].isin(part_samples)].values.tolist()
        for i in range(len(picked_df)):
            row = picked_df[i].copy()
            row.append(sample_to_class[row[0]])
            row.append(sample_to_sim[row[0]])
            table.append(row)
    
    f = pd.DataFrame(table, columns = list(de.head().columns) + ["IMAGENET_CLASS", "CLASS_SIM"])
    df.to_parquet('collection.snappy.parquet')