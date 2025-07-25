import numpy as np
import re
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.stats import entropy


class Qda:
    def __init__(self):
        self.to_drop = []  # Инициализация параметра класса to_drop
        self.qda = None  # Инициализация параметра класса qda

    def read_result(self,mydata_path):
    # Получаем список всех файлов в папке
        files = os.listdir(mydata_path)

        data= np.empty((0,121*25), float)  # Создаем пустой 2D массив с 3 столбцами

        # Добавляем строки


        # Проходимся по файлам
        for file in files:
            file_path = os.path.join(mydata_path, file)  # Полный путь к файлу
            # Путь к файлу
            #file_path = "./p/877P_12_09_31.09.txt"

        # Читаем файл и обрабатываем данные
            data_1 = []
            start_reading = False

            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    # Если найдена строка Data Points, начинаем считывать данные
                    if "Data Points" in line:
                        start_reading = True
                        continue
                    if start_reading:
                        # Разделяем строку по пробелам или табуляции и добавляем в список
                        values = re.split(r'\s+', line.strip())
                        if values:  # Проверяем, что строка не пустая
                            data_1.append([float(v) for v in values])

            # Преобразуем список в numpy-массив
            data_array = np.array(data_1)

            # Выводим первые 5 строк массива для проверки
            print(data_array[:])
            print(data_array.shape)

            data_array_1 = data_array.flatten('F')[121:]

            print(data_array_1[:])
            print(data_array_1.shape)
            data = np.vstack((data, data_array_1))

        print(data[:])
        print(data.shape)

        df = pd.DataFrame(data)

        print(df)

        return df, files




    def data_graf(self,df):
        segment_size = 121*25
        num_segments = int(3025/segment_size)  # 3025 // 121 = 25

        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size
            df_segment = df.iloc[:, start:end]

        # Преобразуем DataFrame в длинный формат для Seaborn
            df_long = df_segment.reset_index().melt(id_vars="index", var_name="Column", value_name="Value")

        # Создаем новый рисунок для каждого участка
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=df_long, x="Column", y="Value", hue="index")

        # Подписи графика
            plt.title(f"График {i+1} (столбцы {start}-{end-1})")
            plt.xlabel("Столбцы")
            plt.ylabel("Значения")
            plt.legend(title="Строки")
        
        # Отображаем график
            plt.show(block=False)



    def data_corr(self,df):
        selected_cols = [i*121 for i in range(25)]  
        corr_matrix = df[selected_cols].corr()


    # Визуализируем
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Матрица корреляций")
        plt.show(block=False)

    #folder_path = "./p"

    def view_data(self,mydata_path):
        df,_ =self.read_result(mydata_path)
        self.data_graf(df)
        self.data_corr(df)
        return df




    def data_graf2(self,df1,df2):
        segment_size = 121*25
        num_segments = int(3025/segment_size)  # 3025 // 121 = 25

        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size
            df1_segment = df1.iloc[:, start:end]
            df2_segment = df2.iloc[:, start:end]

        # Преобразуем DataFrame в длинный формат для Seaborn
            df1_long = df1_segment.reset_index().melt(id_vars="index", var_name="Column", value_name="Value")
            df2_long = df2_segment.reset_index().melt(id_vars="index", var_name="Column", value_name="Value")


        # Создаем новый рисунок для каждого участка
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=df1_long, x="Column", y="Value" , hue="index",palette="Blues")
            sns.lineplot(data=df2_long, x="Column", y="Value", hue="index",palette="Reds")

        # Подписи графика
            plt.title(f"График {i+1} (столбцы {start}-{end-1})")
            plt.xlabel("Столбцы")
            plt.ylabel("Значения")
            plt.legend(title="Строки")
        
        # Отображаем график
            plt.show()

    #data_graf2(df_p,df_n)


    def QDAanalysis(self,folder_pass_path,folder_fail_path):

        df_p=self.view_data(folder_pass_path)

        df_n=self.view_data(folder_fail_path)

        df_p1=df_p.copy()
        df_n1=df_n.copy()

        df_p1["dataset"]="1"
        df_n1["dataset"]="0"

        #print(df_p)
        #print(df_n)

        df_combined = pd.concat([df_p1, df_n1])

        print(df_combined)

        # df_combined = pd.concat([df_p, df_n])

        # segment_size = 121*25
        # num_segments = int(3025/segment_size)  # 3025 // 121 = 25

        # start = 0
        # end = start + segment_size
        # df_segment = df_combined.iloc[:, start:end]


        #     # Преобразуем DataFrame в длинный формат для Seaborn
        # df_long = df_segment.reset_index().melt(id_vars="index", var_name="Column", value_name="Value")

        # print(df_long)

        # Создаем палитру с красными и синими тонами
        palette = {0: "red", 1: "blue"}

        # Строим график
        df_c_long = df_combined.reset_index().melt(id_vars=["index","dataset"], var_name="Column", value_name="Value")
        #df_n_long = df_n.reset_index().melt(id_vars="index", var_name="Column", value_name="Value")
        sns.lineplot(data=df_c_long, x="Column", y="Value",hue='dataset',units="index",
            estimator=None)
        #sns.lineplot(data=df_n_long, x="Column", y="Value")



        #     # Подписи графика
        # plt.title(f"График {i+1} (столбцы {start}-{end-1})")
        # plt.xlabel("Столбцы")
        # plt.ylabel("Значения")
        # plt.legend(title="Строки")
            
        #     # Отображаем график
        plt.show(block=False)
        # #plt.show()

        #----------------------------QDA--QDA--QDA--QDA--QDA--QDA--------------------------
        print('QDA')

        X = df_combined.drop(columns=['dataset'])  # Признаки
        y = df_combined['dataset']

        # 3. Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


        corr_matrix = X_train.corr().abs()  # Вычисляем корреляцию по модулю
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.90)]

        print(f'to_drop: {self.to_drop}')


        X_train = X_train.drop(columns=self.to_drop)
        X_test = X_test.drop(columns=self.to_drop)

        print(X_train.shape)
        # ==== 3. Применяем PCA ====
        # n_components = 80  # Можно варьировать 100-500
        # pca = PCA(n_components=n_components)

        # X_train_pca = pca.fit_transform(X_train)  # Обучаем PCA на train
        # X_test_pca = pca.transform(X_test)  # Применяем к test

        # ==== 4. Обучаем QDA ====
        self.qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
        self.qda.fit(X_train, y_train)

        # 5. Предсказание и оценка точности
        y_pred = self.qda.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        qda_probs = self.qda.predict_proba(X_test)
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
        return self.qda

    def QDAfit(self,folder_path):

        df_fd=self.view_data(folder_path)

        df_fd1=df_fd.copy()
     

        df_p1["dataset"]="1"
        df_n1["dataset"]="0"

        #print(df_p)
        #print(df_n)

        df_combined = pd.concat([df_p1, df_n1])

        print(df_combined)



        plt.show(block=False)


        #----------------------------QDA--QDA--QDA--QDA--QDA--QDA--------------------------
        print('QDA')

        X = df_combined.drop(columns=['dataset'])  # Признаки
        y = df_combined['dataset']


        X_train = X_train.drop(columns=self.to_drop)
        X_test = X_test.drop(columns=self.to_drop)

        print(X_train.shape)

        qda.fit(X_train, y_train)

    def classify_files(self, folder_path ):
        """
        Классифицирует файлы в заданном каталоге на основе обученной модели QDA.

        Args:
        folder_path (str): Путь к каталогу с файлами для анализа.
        qda_model (QuadraticDiscriminantAnalysis): Обученная модель QDA.
        to_drop (list): Список признаков, которые нужно исключить из данных.

        Returns:
        dict: Словарь, где ключ — имя файла, значение — признак ('P' или 'N').
        """


            # Читаем данные из файлаfile_pa
        df,files = self.read_result(folder_path)

            # Удаляем ненужные признаки
        df = df.drop(columns=self.to_drop, errors='ignore')

            # Прогнозируем метку с помощью модели QDA
        predictions = self.qda.predict(df)
           #majority_class = 'P' if predictions.mean() > 0.5 else 'N'
            # Сохраняем результат в словарь
        classification_results = {file: prediction for file, prediction in zip(files, predictions)}
      
        return classification_results


if __name__=='__main__':
    qda=Qda()
    folder_pass_path = "./p"
    folder_fail_path = "./n"
    qda.QDAanalysis(folder_pass_path,folder_fail_path)