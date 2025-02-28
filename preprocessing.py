from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def select_data():
    df_attr = pd.read_csv("data/list_attr_celeba.csv")
    df_partition = pd.read_csv("data/list_eval_partition.csv")
    
    df = pd.merge(df_partition, df_attr, on="image_id")
    df = df[["image_id", "Male", "partition"]]
    df["Male"] = df["Male"].apply(lambda x: 1 if x == 1 else 0)

    img_src_folder = Path("data/img_align_celeba/img_align_celeba")
    img_dest_folder = Path("data/dataset")

    keep_images = set(df[df["partition"] == 1]["image_id"])

    # Copier les images avec pathlib
    for file in tqdm(keep_images, desc=f"Copy images", unit="img"):
        src_path = img_src_folder / file
        dest_path = img_dest_folder / file
        if src_path.exists():
            dest_path.write_bytes(src_path.read_bytes())

    df = df[df["partition"] == 1]
    df.to_csv("data/data.csv", index=False)

    print(f"{len(df)} images copiées")
    
    
    


def load_data_RFC(csv_path, img_folder):
    df = pd.read_csv(csv_path)
    
    X, y = [], []

    for _, row in tqdm(df.iterrows(), desc=f"Processing images", unit="img"):
        file = row["image_id"].replace(".jpg", "") + ".npy"
        label = row["Male"]  

        img_path = os.path.join(img_folder, file)

        if not os.path.exists(img_path):
            print(f"Fichier manquant : {img_path}")
            continue  

        img = np.load(img_path).astype(np.float32)
        img = img.flatten()  

        X.append(img)
        y.append(label)
        
    return np.array(X), np.array(y)


def split_data_DNN(csv_path, test_ratio=0.2, random_state=42):
        df = pd.read_csv(csv_path)
        
        np.random.seed(random_state)  
        shuffled_indices = np.random.permutation(len(df))

        test_size = int(len(df) * test_ratio)
        test_indices = shuffled_indices[:test_size]
        train_indices = shuffled_indices[test_size:]

        train = df.iloc[train_indices]
        test = df.iloc[test_indices]
        
        print("Split fait")
        
        return train, test