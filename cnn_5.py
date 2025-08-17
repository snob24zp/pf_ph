# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 14:10:49 2025

@author: R
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.signal import savgol_filter

# ===================== Dataset ===================== #
class SensorDataset(Dataset):
    def __init__(self, signals, labels, max_len=140):
        """
        signals: list/array [N, 12, T] (T = 100..140)
        labels: list/array [N]
        """
        self.max_len = max_len
        self.data, self.mask = [], []
        self.labels = labels

        for sig in signals:
            # pad or crop
            T = sig.shape[1]
            arr = np.zeros((12, max_len))
            msk = np.zeros(max_len)
            if T <= max_len:
                arr[:, :T] = sig
                msk[:T] = 1
            else:
                arr[:, :max_len] = sig[:, :max_len]
                msk[:] = 1

            # производные Savitzky–Golay
            d1 = savgol_filter(arr, 7, 3, deriv=1, axis=1)
            d2 = savgol_filter(arr, 7, 3, deriv=2, axis=1)

            # stack: [36, max_len]
            arr = np.concatenate([arr, d1, d2], axis=0)

            self.data.append(arr.astype(np.float32))
            self.mask.append(msk.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx]),      # [36, max_len]
            torch.tensor(self.mask[idx]),      # [max_len]
            torch.tensor(self.labels[idx]).float()
        )

# ===================== Blocks ===================== #
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        ks = [3, 5, 9, 15]
        self.branches = nn.ModuleList()
        for k in ks:
            self.branches.append(
                nn.Conv1d(in_ch, out_ch // len(ks), kernel_size=k, padding=k//2)
            )
        # dilated conv
        self.dilated = nn.Conv1d(in_ch, out_ch // len(ks),
                                 kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        outs.append(self.dilated(x))
        return torch.cat(outs, dim=1)

class TCNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm1d(ch)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out + res

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // r)
        self.fc2 = nn.Linear(ch // r, ch)

    def forward(self, x):
        w = x.mean(-1)          # [B, C]
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(-1)

# ===================== Model ===================== #
class SensorNet(nn.Module):
    def __init__(self, in_ch=36, hid=64):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, hid, 3, padding=1)
        self.inc1 = InceptionBlock(hid, hid)
        self.inc2 = InceptionBlock(hid, hid)
        self.tcn = TCNBlock(hid)
        self.se = SEBlock(hid)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hid*2, 1)

    def forward(self, x, mask):
        # x: [B, C, T], mask: [B, T]
        out = F.relu(self.stem(x))
        out = self.inc1(out)
        out = self.inc2(out)
        out = self.tcn(out)
        out = self.se(out)

        mask = mask.unsqueeze(1)
        masked = out * mask

        # global avg & max pooling
        s = mask.sum(-1, keepdim=True) + 1e-6
        avg = masked.sum(-1) / s
        mx = masked.max(-1).values
        feat = torch.cat([avg, mx], dim=1)

        feat = self.dropout(feat)
        logit = self.fc(feat).squeeze(1)
        return logit

# ===================== Focal Loss ===================== #
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1-pt)**self.gamma * bce
        return loss.mean()

# ===================== Training Loop (K-fold CV) ===================== #
def train_cv(signals, labels, n_splits=5, epochs=30, batch_size=16, lr=1e-3, device="cuda"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    for fold, (tr, va) in enumerate(skf.split(signals, labels)):
        print(f"Fold {fold+1}")
        tr_data = SensorDataset([signals[i] for i in tr], labels[tr])
        va_data = SensorDataset([signals[i] for i in va], labels[va])
        tr_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(va_data, batch_size=batch_size)

        model = SensorNet().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = FocalLoss()

        for ep in range(epochs):
            model.train()
            for xb, msk, yb in tr_loader:
                xb, msk, yb = xb.to(device), msk.to(device), yb.to(device)
                logit = model(xb, msk)
                loss = crit(logit, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            preds, gts = [], []
            with torch.no_grad():
                for xb, msk, yb in va_loader:
                    xb, msk = xb.to(device), msk.to(device)
                    logit = model(xb, msk)
                    preds.extend(torch.sigmoid(logit).cpu().numpy())
                    gts.extend(yb.numpy())
            auc = roc_auc_score(gts, preds)
            ap = average_precision_score(gts, preds)
            print(f"  Epoch {ep+1}: ROC-AUC={auc:.3f}, PR-AUC={ap:.3f}")
        results.append((auc, ap))
    return results

# ===================== Пример использования ===================== #
if __name__ == "__main__":
    np.random.seed(42)

    N = 100   # кол-во образцов
    signals, labels = [], []

    for i in range(N):
        T = np.random.randint(100, 141)  # длина записи
        # базовая "шляпа": гауссиана по времени
        t = np.linspace(-2, 2, T)
        base = np.exp(-t**2)
        base = base / base.max()

        # добавляем шум и небольшие "зубцы"
        signal = []
        for ch in range(12):
            s = base + 0.05*np.random.randn(T)
            if np.random.rand() < 0.3:  # иногда зубцы
                spike_pos = np.random.randint(20, T-20)
                s[spike_pos:spike_pos+3] += np.random.rand()*0.5
            signal.append(s)
        signal = np.array(signal)
        signals.append(signal)

        # бинарная метка (например, зависит от «наличия зубца»)
        labels.append(int(np.max(signal) > 1.2))

    signals = np.array(signals, dtype=object)  # массив объектов (разная длина)
    labels = np.array(labels)

    # обучение (можно уменьшить эпохи для теста)
    results = train_cv(signals, labels, n_splits=3, epochs=5, batch_size=8, device="cpu")

    print("CV results:", results)
