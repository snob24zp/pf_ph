import numpy as np
import re
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#models
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.stats import entropy

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt





import pf_read_file2 as pfrf

class ML_model:
    def __init__(self,model,SP):
        self.to_drop = []  # Инициализация параметра класса to_drop
        self.model =  model
        self.SP=SP



    def QDAanalysis(self):


        print('QDA')
        nsensor=12
        data_points=self.SP.data_points
        points=nsensor*data_points
        X=self.SP.df.iloc[:, :points]

        #X = self.SP.df.drop(columns=['dataset'])  # Признаки
        y = self.SP.df['dataset']
        self.X=X
        self.y=y

        print('QDAanalysis X.shape',X.shape)

        # 3. Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


        corr_matrix = X_train.corr().abs()  # Вычисляем корреляцию по модулю
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 1.90)]

        print(f'to_drop: {self.to_drop}')


        X_train = X_train.drop(columns=self.to_drop)
        X_test = X_test.drop(columns=self.to_drop)

        print('X_train.shape',X_train.shape)
        # ==== 3. Применяем PCA ====
        # n_components = 80  # Можно варьировать 100-500
        # pca = PCA(n_components=n_components)

        # X_train_pca = pca.fit_transform(X_train)  # Обучаем PCA на train
        # X_test_pca = pca.transform(X_test)  # Применяем к test

        # ==== 4. Обучаем QDA ====
        self.pipeline = Pipeline([
            ('classifier', self.model)
        ])

        self.pipeline.fit(X_train, y_train)

        # 5. Предсказание и оценка точности
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        qda_probs = self.pipeline.predict_proba(X_test)
        #print(f'qda_probs: {qda_probs:.2f}')
        print('qda_probs')
        print(qda_probs)
        confidence_scores = np.max(qda_probs, axis=1) 
        print('confidence_scores')
        print(confidence_scores)

        uncertainty_scores = entropy(qda_probs.T)
        print('uncertainty_scores')
        print(uncertainty_scores)


        print(f'Accuracy: {accuracy:.2f}')
        
        # === 6. Финальный Pipeline ===
        
        # === 7. Кросс-валидация ===
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X=self.X.drop(columns=self.to_drop)
        cv_results = cross_validate(
            self.pipeline, X, y,
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

        
        return self.model
    
    def learn_curv(self):
       
        X=self.X.drop(columns=self.to_drop) 
        print('learn_curv X.shape',X.shape)
        train_sizes, train_scores, test_scores = learning_curve(
        estimator=self.model,
        X=X, y=self.y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='balanced_accuracy',
        n_jobs=-1
        )

        # %% Обробка результатів
        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        test_mean  = test_scores.mean(axis=1)
        test_std   = test_scores.std(axis=1)
        
        # %% Візуалізація
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label="Train", marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
        
        plt.plot(train_sizes, test_mean, label="Validation", marker='s')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
        
        plt.title("Крива навчання (Balanced Accuracy)", fontsize=14)
        plt.xlabel("Кількість навчальних зразків")
        plt.ylabel("Balanced Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        print('train_sizes=np.linspace(0.1, 1.0, 10)',np.linspace(0.1, 1.0, 10),'train_mean',train_mean,'test_mean',test_mean)

        return
    

# =============================================================================
#     def QDAfit(self,folder_path):
# 
#         print('QDA')
# 
#         X = df_combined.drop(columns=['dataset'])  # Признаки
#         y = df_combined['dataset']
# 
# 
#         X_train = X_train.drop(columns=self.to_drop)
#         X_test = X_test.drop(columns=self.to_drop)
# 
#         print(X_train.shape)
# 
#         qda.fit(X_train, y_train)
# 
#     def classify_files(self, folder_path ):
#         """
#         Классифицирует файлы в заданном каталоге на основе обученной модели QDA.
# 
#         Args:
#         folder_path (str): Путь к каталогу с файлами для анализа.
#         qda_model (QuadraticDiscriminantAnalysis): Обученная модель QDA.
#         to_drop (list): Список признаков, которые нужно исключить из данных.
# 
#         Returns:
#         dict: Словарь, где ключ — имя файла, значение — признак ('P' или 'N').
#         """
# 
# 
#             # Читаем данные из файлаfile_pa
#         df,files = self.read_result(folder_path)
# 
#             # Удаляем ненужные признаки
#         df = df.drop(columns=self.to_drop, errors='ignore')
# 
#             # Прогнозируем метку с помощью модели QDA
#         predictions = self.qda.predict(df)
#            #majority_class = 'P' if predictions.mean() > 0.5 else 'N'
#             # Сохраняем результат в словарь
#         classification_results = {file: prediction for file, prediction in zip(files, predictions)}
#       
#         return classification_results
# 
# =============================================================================

if __name__=='__main__':
    Pr=pfrf.ProccesingFFE()

    folder_pass_path = "./p2/p2"
    folder_fail_path = "./n2/n2"
    #folder_pass_path = "./Chicken Data Combined PASS/Chicken Data Combined PASS"
    #folder_fail_path = "./Chicken Data Combined FAIL/Chicken Data Combined FAIL"

    #Pr.view(folder_pass_path,folder_fail_path)
    #Pr.eda(folder_pass_path, folder_fail_path)
    Pr.fe(folder_pass_path, folder_fail_path) 
# =============================================================================
#     qda=ML_model(QuadraticDiscriminantAnalysis(reg_param=0.1),Pr.SP)    
#     qda.QDAanalysis()
#     qda.learn_curv()
#     print("KNeighborsClassifier")
# =============================================================================
    knn=ML_model(KNeighborsClassifier(
    n_neighbors=7,        # количество ближайших соседей
    #metric='cosine',      # косинусная метрика — хорошо работает при такой высокой размерности
    weights='distance',   # ближние соседи важнее (меньше шумов)
    n_jobs=-1             # использовать все ядра процессора
           ),Pr.SP)    
    knn.QDAanalysis()
    knn.learn_curv()