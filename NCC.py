# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.neighbors import NearestCentroid

import pf_read_file2 as pfrf


if __name__ == "__main__":



    # ==== 1. Загрузка данных ====
    Pr = pfrf.ProccesingFFE()
    folder_pass_path = "./p2/p2"
    folder_fail_path = "./n2/n2"
    folder_pass_path = "./Disease_No-Disease Samples 2/Disease_No-Disease Samples 2/NP"
    folder_pass_path = "./Disease_No-Disease Samples 2/Disease_No-Disease Samples 2/P"
    folder_pass_path = "./Disease_No-Disease Samples 1/Disease_No-Disease Samples 1/NP"
    folder_pass_path = "./Disease_No-Disease Samples 1/Disease_No-Disease Samples 1/P"
    Pr.fe2(folder_pass_path, folder_fail_path)
    
    df=Pr.SP.df
    
    X = df.iloc[:,0:Pr.SP.data_points*(Pr.SP.sensor_number-1)]
    y = df['dataset']
    
    print(X)
    print(Pr.SP.data_points,Pr.SP.sensor_number,X.shape)
    print(y)
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
 
    # y_preds=nearest_centroid_classifier(X_train, y_train, X_test)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = NearestCentroid()
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=['accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'precision', 'recall'],
        return_train_score=True,
        n_jobs=-1
    )
    
    # === 8. Вывод результатов ===
    print("=== Крос-валідація ===")
    for metric in ['accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'precision', 'recall']:
        print(f"Train {metric}: {np.mean(cv_results['train_' + metric]):.4f} ± {np.std(cv_results['train_' + metric]):.4f}")
        print(f"Test  {metric}: {np.mean(cv_results['test_' + metric]):.4f} ± {np.std(cv_results['test_' + metric]):.4f}")
        print("---")


    # # ==== 2. Подготовка ====
    # dataset = SensorDataset(Pr.SP)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    
    # generator = torch.Generator().manual_seed(42)
    # train_set, val_set = random_split(
    #                      dataset, [train_size, val_size], generator=generator)
    # train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=8)

    # # ==== 3. Обучение ====
    # model = CNN1D().to(device)  # ==== ИЗМЕНЕНИЕ ====
    # train(model, train_loader, val_loader)
    
    # accuracy, precision, recall, f1,wrong_indices = evaluate_model(
    #    model, val_loader, device, print_label=True)  # ==== ИЗМЕНЕНИЕ ====
    # print(f"Accuracy (val_loader): {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


    # fname_list = sorted(Pr.SP.df.iloc[wrong_indices]['fname'].tolist())
    # print(fname_list)
   
    # accuracy, precision, recall, f1,wrong_indices = evaluate_model(
    #    model, train_loader, device, print_label=True)  # ==== ИЗМЕНЕНИЕ ====
    # print(f"Accuracy (train_loader): {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # fname_list = sorted(Pr.SP.df.iloc[wrong_indices]['fname'].tolist())
    # print(fname_list)


    # # ==== 6. K-Fold для стабильности ====
    # k_fold_training(dataset, CNN1D, k=5, epochs=50, batch_size=8, lr=5e-5, device='cpu')
