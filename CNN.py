# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 18:21:28 2025

@author: i5
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

import pf_read_file2 as pfrf

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy



# =============================================================================
# # ==== 1. Пример генерации искусственных данных (замени на реальные) ====
# # shape: (100, 12, 121)
# X = np.random.randn(100, 12, 121).astype(np.float32)
# y = np.random.randint(0, 2, size=(100,)).astype(np.longlong)
# 
# =============================================================================

# ==== 2. Кастомный Dataset ====
class SensorDataset(Dataset):
    def __init__(self, SP):
        nsensor=12
        data_points=SP.data_points
        points=nsensor*data_points
        df_12=SP.df.iloc[:, :points]
        N = SP.df.shape[0]  # количество строк (образцов)

        arr = df_12.values.reshape(N, 12, data_points)  # reshape с динамическим размером батча
        tensorX = torch.tensor(arr, dtype=torch.float32)
        y=SP.df["dataset"]
        self.X = tensorX
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# ==== 3. Модель CNN ====
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.global_pool = nn.AdaptiveAvgPool1d(5)  # адаптивное усреднение в 5 сегментов
        self.dropout = nn.Dropout(0.4)

        # 128 каналов × 5 временных сегментов → 640 входов в FC
        self.fc1 = nn.Linear(128 * 5, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))  # → (B, 64, L/2)
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))  # → (B, 128, L/4)
        x = self.global_pool(x)                              # → (B, 128, 5)
        x = x.view(x.size(0), -1)                            # → (B, 128×5)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))                          # → (B, 128)
        return torch.sigmoid(self.fc2(x)).squeeze(dim=1)          # → (B,)   

# ==== 4. Обучение ====
def train(model, train_loader, val_loader, epochs=20, lr=1e-4):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
# =============================================================================
#     best_model_wts = None
#     best_val_acc = 0.0
#     train_losses, val_losses = [], []
#     train_accuracies, val_accuracies = [], []
#     
#     scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
# 
# =============================================================================
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch, y_batch.float()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

        # Валидация
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                y_pred = model(X_val) > 0.5
                correct += (y_pred == y_val).sum().item()
                total += y_val.size(0)
        acc = correct / total
        print(f"Validation Accuracy: {acc:.2f}")
        

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            preds = (outputs > 0.5).long()  # бинарный threshold
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return accuracy, precision, recall, f1


from sklearn.model_selection import KFold
from torch.utils.data import Subset


def k_fold_training(dataset, model_class, k=5, epochs=50, batch_size=8, lr=1e-4, device='cpu'):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n====== Fold {fold + 1} / {k} ======")

        # Разбиение данных
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        # Инициализация модели
        model = model_class().to(device)

        # Обучение
        train(model, train_loader, val_loader, epochs=epochs, lr=lr)

        # Оценка
        metrics = evaluate_model(model, val_loader, device)
        all_metrics.append(metrics)
        print(f"Fold {fold + 1} metrics: Accuracy={metrics[0]:.4f}, Precision={metrics[1]:.4f}, Recall={metrics[2]:.4f}, F1={metrics[3]:.4f}")

    # Средние метрики по всем фолдам
    all_metrics = np.array(all_metrics)
    avg = all_metrics.mean(axis=0)
    print(f"\n📊 Average over {k} folds:\n"
          f"Accuracy={avg[0]:.4f}, Precision={avg[1]:.4f}, Recall={avg[2]:.4f}, F1={avg[3]:.4f}")

if __name__=="__main__":
    print(torch.__version__)         # должна быть >2.0
    print(torch.cuda.is_available()) # будет False — это нормально
    Pr=pfrf.ProccesingFFE()
    folder_pass_path = "./p2/p2"
    folder_fail_path = "./n2/n2"
    #folder_pass_path = "./Chicken Data Combined PASS/Chicken Data Combined PASS"
    #folder_fail_path = "./Chicken Data Combined FAIL/Chicken Data Combined FAIL"

    #Pr.view(folder_pass_path,folder_fail_path)
    #Pr.eda(folder_pass_path, folder_fail_path)
    Pr.fe(folder_pass_path, folder_fail_path)

    # ==== 5. Подготовка ====
    dataset = SensorDataset(Pr.SP)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)
    
    model = CNN1D()
    train(model, train_loader, val_loader)
    
    accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    

    k_fold_training(dataset, CNN1D, k=5, epochs=200, batch_size=8, lr=5e-5, device='cpu')