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
import pandas as pd

device='cpu'

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
        df_0=SP.df.iloc[:, :points]
        #print('SensorDataset',df_0,df_0.shape)
        df_49=SP.df.iloc[:, SP.data_points*49:SP.data_points*(49+12)]
        #print('SensorDataset',df_49,df_49.shape)
        df_data = pd.concat([df_0, df_49],axis=1, ignore_index=True)
        #print('SensorDataset',df_data,df_data.shape)
        N = SP.df.shape[0]  # количество строк (образцов)

        arr = df_data.values.reshape(N, 24, data_points)  # reshape с динамическим размером батча
        tensorX = torch.tensor(arr, dtype=torch.float32)
        #print('dataset',tensorX)
        y=SP.df["dataset"]
        self.X = tensorX
        self.y = torch.tensor(y.values, dtype=torch.float32)
        
        print('SensorDataset',self.y.min(), self.y.max())
        print(self.y.unique())
        
        is_sequential = SP.df.index.is_monotonic_increasing and SP.df.index.equals(pd.RangeIndex(len(SP.df)))

        print("Индексы идут по порядку без пропусков:", is_sequential)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# ==== 3. Модель CNN ====
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(24, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.global_pool = nn.AdaptiveMaxPool1d(20)
        self.dropout_conv = nn.Dropout(0.1)

        self.fc1 = nn.Linear(128 * 20, 128)
        self.dropout_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return torch.sigmoid(self.fc2(x)).squeeze(dim=1)

# ==== 4. Обучение ====
def train(model, train_loader, val_loader, epochs=50, lr=5e-5):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 🔄 Добавлено: списки для графиков
    train_losses = []
    val_accuracies = []

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

        avg_loss = total_loss / len(train_loader)  # 🔄 Добавлено
        train_losses.append(avg_loss)              # 🔄 Добавлено
        
        #print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # === Валидация ===
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                y_pred = model(X_val) > 0.5
                correct += (y_pred == y_val).sum().item()
                total += y_val.size(0)

        acc = correct / total
        val_accuracies.append(acc)  # 🔄 Добавлено
# =============================================================================
#         print(f"Validation Accuracy: {acc:.2f}")
#         
#         accuracy, precision, recall, f1,wrong_indices = evaluate_model(
#            model, val_loader, device, print_label=True)  # ==== ИЗМЕНЕНИЕ ====
#         print(f"Accuracy (val_loader): {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
# 
#         accuracy, precision, recall, f1,wrong_indices = evaluate_model(
#            model, train_loader, device, print_label=True)  # ==== ИЗМЕНЕНИЕ ====
#         print(f"Accuracy (train_loader): {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
# 
# =============================================================================

    # 🔄 Добавлено: визуализация графиков
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, marker='o', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

def train(model, train_loader, val_loader, epochs=50, lr=5e-5):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []           # Для потерь на валидации
    val_accuracies = []

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

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # === Валидация ===
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val, y_val.float()
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()

                y_pred = outputs > 0.5
                correct += (y_pred == y_val).sum().item()
                total += y_val.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        acc = correct / total
        val_accuracies.append(acc)

# =============================================================================
#         print(f"Validation Accuracy: {acc:.2f}")
#         
#         accuracy, precision, recall, f1,wrong_indices = evaluate_model(
#            model, val_loader, device, print_label=True)  # ==== ИЗМЕНЕНИЕ ====
#         print(f"Accuracy (val_loader): {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
# 
#         accuracy, precision, recall, f1,wrong_indices = evaluate_model(
#            model, train_loader, device, print_label=True)  # ==== ИЗМЕНЕНИЕ ====
#         print(f"Accuracy (train_loader): {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
# 
# =============================================================================

    print('Train Loss',train_losses) 
    print('Validation Loss',val_losses)


    plt.figure(figsize=(10, 5))

    plt.plot(train_losses, marker='o', label='Train Loss')
    plt.plot(val_losses, marker='o', label='Validation Loss', color='red')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Отдельно accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(val_accuracies, marker='o', color='green')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()        

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, data_loader, device,print_label=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs=[]
    all_inputs=[]
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            
            probs = model(inputs)
            preds = (probs > 0.5).long()            
            #-----
            all_inputs.append(inputs.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu().long())

    X_all = torch.cat(all_inputs)
    y_true = torch.cat(all_labels)
    y_pred = torch.cat(all_preds)
    y_prob = torch.cat(all_probs)

    wrong_indices = (y_true != y_pred).nonzero(as_tuple=True)[0]
    if print_label:
        print(f"Ошибок классификации: {len(wrong_indices)} из {len(y_true)}")
        #print(wrong_indices)
        
        # ==== 5. Визуализация ошибок ====  # ==== ИЗМЕНЕНИЕ ====
# =============================================================================
#         for i in range(min(1, len(wrong_indices))):
#             idx = wrong_indices[i]
#             signal = X_all[idx].numpy()
#             true_label = y_true[idx].item()
#             pred_label = y_pred[idx].item()
#             prob = y_prob[idx].item()
# 
#             plt.figure(figsize=(10, 3))
#             for ch in range(signal.shape[0]):
#                 plt.plot(signal[ch], label=f"Sens {ch}", alpha=0.7)
#             plt.title(f"[{i}] True={true_label}, Pred={pred_label}, Prob={prob:.2f}")
#             plt.legend(loc='upper right', ncol=4)
#             plt.tight_layout()
#             plt.show()
# 
# =============================================================================
            
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return accuracy, precision, recall, f1,wrong_indices


from sklearn.model_selection import KFold
from torch.utils.data import Subset


def k_fold_training(dataset, model_class, k=5, epochs=50, batch_size=8, lr=1e-5, device='cpu'):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_metrics = []

    val_set=set()
    wrong_set=set()
    whole_set= set(range(len(dataset)))
    

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
        all_metrics.append(metrics[:-1])
        
        val_set.update(val_idx.tolist())
        wrong_set.update(metrics[-1].tolist())
        
        print(f"Fold {fold + 1} metrics: Accuracy={metrics[0]:.4f}, Precision={metrics[1]:.4f}, Recall={metrics[2]:.4f}, F1={metrics[3]:.4f}")
        
        fname_list = sorted(Pr.SP.df.iloc[metrics[-1]]['fname'].tolist())
        print(fname_list)

        
        accuracy, precision, recall, f1,wrong_indices = evaluate_model(
           model, train_loader, device, print_label=True)  # ==== ИЗМЕНЕНИЕ ====
        print(f"Accuracy (train_loader): {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        fname_list = sorted(Pr.SP.df.iloc[wrong_indices]['fname'].tolist())
        print(fname_list)


    # Средние метрики по всем фолдам
    all_metrics = np.array(all_metrics)
    avg = all_metrics.mean(axis=0)
    print(f"\n📊 Average over {k} folds:\n"
          f"Accuracy={avg[0]:.4f}, Precision={avg[1]:.4f}, Recall={avg[2]:.4f}, F1={avg[3]:.4f}")
    
    non_val_set=whole_set-val_set
    print('val_set',val_set)
    print('non_val_set',non_val_set)
    print('wrong_set',wrong_set)
    fname_list = sorted(Pr.SP.df.iloc[list(wrong_set)]['fname'].tolist())
    print('wrong_set',fname_list)

    
if __name__ == "__main__":


    # ==== 0. Система ====
    print(torch.__version__)         # должна быть >2.0
    print(torch.cuda.is_available()) # False — это нормально

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # ==== ИЗМЕНЕНИЕ ====

    # ==== 1. Загрузка данных ====
    Pr = pfrf.ProccesingFFE()
    folder_pass_path = "./p2/p2"
    folder_fail_path = "./n2/n2"
    folder_pass_path = "./Disease_No-Disease Samples 2/Disease_No-Disease Samples 2/NP"
    folder_pass_path = "./Disease_No-Disease Samples 2/Disease_No-Disease Samples 2/P"
    folder_pass_path = "./Disease_No-Disease Samples 1/Disease_No-Disease Samples 1/NP"
    folder_pass_path = "./Disease_No-Disease Samples 1/Disease_No-Disease Samples 1/P"
    Pr.af(folder_pass_path, folder_fail_path)

    # ==== 2. Подготовка ====
    dataset = SensorDataset(Pr.SP)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(
                         dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    # ==== 3. Обучение ====
    model = CNN1D().to(device)  # ==== ИЗМЕНЕНИЕ ====
    train(model, train_loader, val_loader)
    
    accuracy, precision, recall, f1,wrong_indices = evaluate_model(
       model, val_loader, device, print_label=True)  # ==== ИЗМЕНЕНИЕ ====
    print(f"Accuracy (val_loader): {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


    fname_list = sorted(Pr.SP.df.iloc[wrong_indices]['fname'].tolist())
    print(fname_list)
   
    accuracy, precision, recall, f1,wrong_indices = evaluate_model(
       model, train_loader, device, print_label=True)  # ==== ИЗМЕНЕНИЕ ====
    print(f"Accuracy (train_loader): {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    fname_list = sorted(Pr.SP.df.iloc[wrong_indices]['fname'].tolist())
    print(fname_list)


    # ==== 6. K-Fold для стабильности ====
    k_fold_training(dataset, CNN1D, k=5, epochs=50, batch_size=8, lr=5e-5, device='cpu')
    