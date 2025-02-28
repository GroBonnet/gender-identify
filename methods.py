import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from model import DNN


class GenderRFClassifier:
    def __init__(self, image_size=(64, 64), n_estimators=100, random_state=42):
        self.image_size = image_size
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    
    
    def process_data(self, csv_path = "data/data.csv", image_folder="data/dataset", method='run', test_size=0.2):
        df = pd.read_csv(csv_path)
        if method=='test':
            df = df.iloc[:100]
        X, y = [], []
        
        for _, row in tqdm(df.iterrows(), desc=f"Processing images", unit="img"):
            img_path = os.path.join(image_folder, row['image_id'])
            img = imread(img_path)  # Charger image
            img_resized = resize(img, self.image_size, anti_aliasing=True)
            img_flatten = img_resized.flatten()
            
            X.append(img_flatten)
            y.append(row['Male'])
        
        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        print(f"Données préparées : {X_train.shape[0]} train | {X_test.shape[0]} test")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Modèle entraîné avec succès.")
    
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Rapport de classification:\n", classification_report(y_test, y_pred))
            
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Modèle sauvegardé dans {filepath}")
        
        
        
           
class GenderDNNClassifier():
    def __init__(self, classifier: nn.Module = None, image_size=(64, 64), lr=0.001, num_epochs=10, batch_size=32):
        self.image_size = image_size  
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = classifier if classifier else DNN().to(self.device)
        
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr)
        
        self.train_losses = []
        self.train_accuracies = []
    
    
    def process_data(self, csv_path="data/data.csv", image_folder="data/dataset", method='run', test_size=0.2):
        df = pd.read_csv(csv_path)
        if method == 'test':
            df = df.iloc[:100]  
        X, y = [], []

        for _, row in tqdm(df.iterrows(), desc=f"Processing images", unit="img"):
            img_path = os.path.join(image_folder, row['image_id'])
            img = imread(img_path)
            img_resized = resize(img, self.image_size, anti_aliasing=True)
            img_flatten = img_resized.flatten()
            X.append(img_flatten)
            y.append(row['Male'])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Données préparées : {len(X_train)} train | {len(X_test)} test")
    
    
    def train(self):
        self.classifier.train()
        for epoch in range(self.num_epochs):
            total_loss, correct = 0, 0
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.classifier(imgs)  
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                
            train_loss = total_loss / len(self.train_loader)
            train_acc = correct / len(self.train_loader.dataset)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {total_loss:.4f} - Train Accuracy: {train_acc:.4f}")
            
        self.plot_loss()


    def plot_loss(self):
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Axe Y pour la loss (à gauche)
        ax1.set_xlabel("Épochs")
        ax1.set_ylabel("Train Loss", color="tab:blue")
        ax1.plot(self.train_losses, label="Train Loss", marker="o", linestyle="-", color="tab:blue")
        ax1.tick_params(axis='y', labelcolor="tab:blue")

        # Axe Y pour l'accuracy (à droite)
        ax2 = ax1.twinx()  
        ax2.set_ylabel("Train Accuracy", color="tab:orange")
        ax2.plot(self.train_accuracies, label="Train Accuracy", marker="s", linestyle="--", color="tab:orange")
        ax2.tick_params(axis='y', labelcolor="tab:orange")

        # Ajout du titre et des légendes
        fig.suptitle("Évolution de la Loss et de l'Accuracy")
        ax1.set_xticks(range(len(self.train_losses)))  # Afficher toutes les époques sur l'axe X
        fig.tight_layout()
        plt.savefig("output/output.png")
        

    def evaluate(self):
        self.classifier.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.classifier(imgs)  
                correct += (outputs.argmax(dim=1) == labels).sum().item()
        val_acc = correct / len(self.test_loader.dataset)
        print(f"\U0001F3AF Validation Accuracy: {val_acc:.4f}")
    
    # def save_model(self, filepath):
    #     torch.save(self.state_dict(), filepath)
    #     print(f"Modèle sauvegardé dans {filepath}")





