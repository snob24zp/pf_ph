# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.neighbors import NearestCentroid

import pf_read_file2 as pfrf

class Model:
    def __init__(self):
        self.model = NearestCentroid()
        self.Pr = pfrf.ProccesingFFE()
    def fit(self,pass_dir,not_pass_dir):
        self.Pr.fe2(pass_dir, not_pass_dir)
        df=self.Pr.SP.df
        X = df.iloc[:,0:self.Pr.SP.data_points*(self.Pr.SP.sensor_number-1)]
        y = df['dataset']
        self.model.fit(X,y)
        
    def classify_files(self,analysis_files_dir): 
            # Читаем данные из файлаfile_pa
        self.Pr.DR.read_result(analysis_files_dir,'+')

            # Удаляем ненужные признаки
        X = self.Pr.DR.df_p.iloc[:,0:self.Pr.SP.data_points*(self.Pr.SP.sensor_number-1)]
        files=self.Pr.DR.df_p["fname"]

            # Прогнозируем метку с помощью модели QDA
        predictions = self.model.predict(X)
           #majority_class = 'P' if predictions.mean() > 0.5 else 'N'
            # Сохраняем результат в словарь
        classification_results = {file: prediction for file, prediction in zip(files, predictions)}
      
        return classification_results


if __name__ == "__main__":

    #mc=Model()

    # ==== 1. Загрузка данных ====
    Pr = pfrf.ProccesingFFE()
    folder_pass_path = "./p2/p2"
    folder_fail_path = "./n2/n2"
    folder_pass_path = "./Disease_No-Disease Samples 2/Disease_No-Disease Samples 2/NP"
    folder_pass_path = "./Disease_No-Disease Samples 2/Disease_No-Disease Samples 2/P"
    folder_fail_path = "./Disease_No-Disease Samples 1/Disease_No-Disease Samples 1/NP"
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

  