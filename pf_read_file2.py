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
import plotly.express as px

import webbrowser

class DataFileFormatError(Exception):
    pass

# читает любые файлы отчёта
# отчёты состоят из данных и параметров
# проверяет однотипность структуры
class DataRead:
    def __init__(self):
        self.data_headers=None
        self.data_headers_number=np.nan
        self.data_points=np.nan
        self.data=None
        self.df_p=None
        self.df_n=None
        
        self.param_dict = {}

    def smart_cast(self,val):
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val  # строка


    def read_result(self,mydata_path, result_type):
        self.data=None
    # Получаем список всех файлов в папке
        files = os.listdir(mydata_path)

        #data= np.empty((0,121*25), float)  # Создаем пустой 2D массив с 3 столбцами

        # Добавляем строки


        # Проходимся по файлам
        for file in files:
            file_path = os.path.join(mydata_path, file)  # Полный путь к файлу
            # Путь к файлу
            #file_path = "./p/877P_12_09_31.09.txt"

        # Читаем файл и обрабатываем данные
            data_1 = []
            start_reading = False
            
            point_count=0
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    
                    # Если найдена строка Data Points, начинаем считывать данные
                    if "Data Points" in line:
                        start_reading = True
                        data_headers1=re.split(r'\s+', line.strip())
                        if not self.data_headers:
                            self.data_headers=data_headers1
                        elif self.data_headers!=data_headers1:
                            DataFileFormatError("different data headers")
                        continue
                        
                    if start_reading:
                        # Разделяем строку по пробелам или табуляции и добавляем в список
                        values = re.split(r'\s+', line.strip())
                        print(values)
                        if values:  # Проверяем, что строка не пустая
                            data_1.append([float(v) for v in values])
                            if np.isnan(self.data_headers_number):
                                self.data_headers_number=len(data_1[0])
                            elif self.data_headers_number!=len(data_1[0]):
                                DataFileFormatError("different data number")
                            
                            point_count=point_count+1
                        
                    elif '=' in line:
                        key, value = [x.strip() for x in line.split('=', 1)]
            
                        # Преобразуем ключи к нужному формату и сохраняем значения
                        pv=self.smart_cast(value)    
                        if self.data is None:#признак что читается перввый файл
                            self.param_dict[key] = pv
                        elif self.param_dict[key]!=pv:
                            DataFileFormatError("different parameters")
                        
            #end of file reading        
            if np.isnan(self.data_points):
                self.data_points=point_count
            elif self.data_points!=point_count:
                DataFileFormatError("different number of data points")
                        
                        
                   
                       
            # Преобразуем список в numpy-массив
            data_array = np.array(data_1)

            # Выводим первые 5 строк массива для проверки
            print(data_array[:])
            print(data_array.shape)

            data_array_1 = data_array.flatten('F')[self.data_points:]

            print(data_array_1[:])
            print(data_array_1.shape)
            
            if  self.data is None:
                self.data = np.copy(data_array_1)
            else:
                self.data = np.vstack((self.data, data_array_1))

        print(self.data[:])
        print(self.data.shape)
        print(self.data_headers)
        df = pd.DataFrame(self.data)
        #df.columns=self.data_headers[2:2+self.data_headers_number]

        print(df)
        accepted_types_p = {'+', 'y', 'pass', 'p'}
        accepted_types_n = {'-', 'n', 'fail', 'f'}

        print("Result type is accepted.")
        if result_type in accepted_types_p:
            self.df_p=df
        elif result_type in accepted_types_n:
            self.df_n=df
        else: DataFileFormatError("unknown result type")
        return df, files

# используеь дополнильные предположения об определённых параметрах    
class Some_Processor:
    def __init__(self):
        self.periods=None
        self.df=None
    def get_params(self,DR):
        self.param_dict=DR.param_dict.copy()
        self.periods = [
        ("Baseline", self.param_dict['Baseline']),
        ("Absorb", self.param_dict['Absorb']),
        ("Pause", self.param_dict['Pause']),
        ("Desorb", self.param_dict['Desorb']),
        ("Flush", self.param_dict['Flush'])           
        ]
        

    def del_pause(self,phase,df=None):
        
        if df is not None:
            self.df=df.copy()
            
        to_drop=[]
        # Преобразуем в количество точек
        segment_lengths = [(name, int(round(seconds * 
            self.param_dict['Acquired data point per second']))) 
                           for name, seconds in self.periods]
        segment_lengths[-1]=(segment_lengths[-1][0],segment_lengths[-1][1]+1)
        # Вычисляем позиции границ
        current_pos = 0
        for sensor in range(24):
            for name, length in segment_lengths:
                if name==phase:
                    to_drop.extend(range(current_pos, current_pos + length))
                current_pos += length
        #df_cleaned = df.drop(to_drop).reset_index(drop=True)
        #df_cleaned = df.drop(df.index[to_drop]).reset_index(drop=True)
        df_cleaned = self.df.drop(self.df.columns[to_drop], axis=1)
        self.df=df_cleaned
        self.periods = [p for p in self.periods if p[0] != phase]
        return df_cleaned
    
    def combo_result(self,dfp,dfn):
        df_p1=dfp.copy()
        df_n1=dfn.copy()

        df_p1["dataset"]=1
        df_n1["dataset"]=0

        #print(df_p)
        #print(df_n)

        df_combined = pd.concat([df_p1, df_n1])

        print("df_combined")
        self.df=df_combined
        return df_combined



    
class Data_Show2:
    def Data_show(self,SP):

        
        df_c_long = SP.df.reset_index().melt(
        id_vars=["index", "dataset"], 
        var_name="Column", 
        value_name="Value"
        )

        # Преобразуем column в строку, если нужно (Plotly лучше работает)
        df_c_long["Column"] = df_c_long["Column"].astype(str)
        
# =============================================================================
#         df_c_long["Column"] = pd.Categorical(
#         df_c_long["Column"],
#         categories=[str(col) for col in df_combined.columns if col != "dataset"],
#         ordered=True
#         )
# 
# =============================================================================

        # Преобразование dataset в строку для корректной окраски
        df_c_long["dataset"] = df_c_long["dataset"].astype(str)
        
        print(df_c_long["dataset"].value_counts())
        print("Уникальные значения:", df_c_long["dataset"].unique())
        print("Уникальные значения:", df_c_long["Column"].unique())
        
        fig = px.line(
            df_c_long, 
            x="Column", 
            y="Value", 
            color="dataset", 
            line_group="index",
            title="Интерактивный график по классам",
            color_discrete_map={"0": 'red', "1": 'blue'},
            category_orders={"dataset": ["0", "1"]}
        )
      
        fig.update_layout(
            hovermode='closest',
            xaxis_title='Признаки',
            yaxis_title='Значения',
            height=600,
            width=1000
        )

#line for phases        
        periods = SP.periods        
        
        # Преобразуем в количество точек
        segment_lengths = [(name, int(round(seconds * 
            SP.param_dict['Acquired data point per second'])))
            for name, seconds in periods]
        segment_lengths[-1]=(segment_lengths[-1][0],segment_lengths[-1][1]+1)
        # Вычисляем позиции границ
        boundaries = []
        current_pos = 0
        for sensor in range(24):
            for name, length in segment_lengths:
                boundaries.append((current_pos, name+' '+str(sensor)))
                current_pos += length
        print("boundaries",boundaries)    
        # Получаем уникальные колонки в правильном порядке
        #columns_sorted = sorted(df_c_long["Column"].unique(), key=lambda x: int(x))
        #columns_sorted = sorted(df_c_long["Column"].unique())
        #print("columns_sorted",columns_sorted) 
        # Сопоставляем позиции точек с колонками
        vlines = []
        for pos, label in boundaries:
            #if pos >= len(columns_sorted):
            #    break
            #col_name = columns_sorted[pos]
            col_name = str(pos)
            print("col_name, label",type(col_name),col_name, label)
            vlines.append((col_name, label))


        shapes = []
        annotations = []
        
        for col, label in vlines:
            shapes.append(dict(
                type="line",
                x0=col, x1=col,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="black", width=1, dash="dot")
            ))
            print("col, label",type(col), col, type(label), label)
            annotations.append(dict(
                x=col,
                y=1.05,
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(size=10, color="black"),
                xanchor="left",
                textangle=-90
            ))

        fig.update_layout(
            shapes=shapes,
            annotations=annotations,
            margin=dict(t=80)  # немного увеличить верхний отступ
        )

        
        # Показываем график в браузере
        fig.write_html("interactive_plot.html")
        webbrowser.open("interactive_plot.html")






class ProccesingFFE:
    def Proccesing(self,folder_pass_path,folder_fail_path):
        DR=DataRead()
        DSh=Data_Show2()
        SP=Some_Processor()
        DR.read_result(folder_pass_path,'+')
        DR.read_result(folder_fail_path,'-')
        SP.get_params(DR)
        SP.combo_result(DR.df_p,DR.df_n)
        dfcс= SP.del_pause("Pause")
        DSh.Data_show(SP)

if __name__=='__main__':
    pr=ProccesingFFE()
    folder_pass_path = "./p"
    folder_fail_path = "./n"
    pr.Proccesing(folder_pass_path, folder_fail_path)

# =============================================================================
# class DataShow1:
# 
#     def data_graf(self,df):
#         segment_size = 121*25
#         num_segments = int(3025/segment_size)  # 3025 // 121 = 25
# 
#         for i in range(num_segments):
#             start = i * segment_size
#             end = start + segment_size
#             df_segment = df.iloc[:, start:end]
# 
#         # Преобразуем DataFrame в длинный формат для Seaborn
#             df_long = df_segment.reset_index().melt(id_vars="index", var_name="Column", value_name="Value")
# 
#         # Создаем новый рисунок для каждого участка
#             plt.figure(figsize=(10, 5))
#             sns.lineplot(data=df_long, x="Column", y="Value", hue="index")
# 
#         # Подписи графика
#             plt.title(f"График {i+1} (столбцы {start}-{end-1})")
#             plt.xlabel("Столбцы")
#             plt.ylabel("Значения")
#             plt.legend(title="Строки")
#         
#         # Отображаем график
#             plt.show(block=False)
# 
# 
# 
#     def data_corr(self,df):
#         selected_cols = [i*121 for i in range(25)]  
#         corr_matrix = df[selected_cols].corr()
# 
# 
#     # Визуализируем
#         plt.figure(figsize=(6, 4))
#         sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
#         plt.title("Матрица корреляций")
#         plt.show(block=False)
# 
#     #folder_path = "./p"
# 
#     def view_data(self,mydata_path):
#         df,_ =self.read_result(mydata_path)
#         self.data_graf(df)
#         self.data_corr(df)
#         return df
# 
# 
# 
# 
#     def data_graf2(self,df1,df2):
#         segment_size = 121*25
#         num_segments = int(3025/segment_size)  # 3025 // 121 = 25
# 
#         for i in range(num_segments):
#             start = i * segment_size
#             end = start + segment_size
#             df1_segment = df1.iloc[:, start:end]
#             df2_segment = df2.iloc[:, start:end]
# 
#         # Преобразуем DataFrame в длинный формат для Seaborn
#             df1_long = df1_segment.reset_index().melt(id_vars="index", var_name="Column", value_name="Value")
#             df2_long = df2_segment.reset_index().melt(id_vars="index", var_name="Column", value_name="Value")
# 
# 
#         # Создаем новый рисунок для каждого участка
#             plt.figure(figsize=(10, 5))
#             sns.lineplot(data=df1_long, x="Column", y="Value" , hue="index",palette="Blues")
#             sns.lineplot(data=df2_long, x="Column", y="Value", hue="index",palette="Reds")
# 
#         # Подписи графика
#             plt.title(f"График {i+1} (столбцы {start}-{end-1})")
#             plt.xlabel("Столбцы")
#             plt.ylabel("Значения")
#             plt.legend(title="Строки")
#         
#         # Отображаем график
#             plt.show()
# 
#     #data_graf2(df_p,df_n)
# 
# =============================================================================

    
# =============================================================================
# 
# 
#         
#         
#     def QDAanalysis(self,folder_pass_path,folder_fail_path):
# 
#         df_p=self.view_data(folder_pass_path)
# 
#         df_n=self.view_data(folder_fail_path)
# 
#         df_p1=df_p.copy()
#         df_n1=df_n.copy()
# 
#         df_p1["dataset"]=1
#         df_n1["dataset"]=0
# 
#         #print(df_p)
#         #print(df_n)
# 
#         df_combined = pd.concat([df_p1, df_n1])
# 
#         print(df_combined)
# 
#         # df_combined = pd.concat([df_p, df_n])
# 
#         # segment_size = 121*25
#         # num_segments = int(3025/segment_size)  # 3025 // 121 = 25
# 
#         # start = 0
#         # end = start + segment_size
#         # df_segment = df_combined.iloc[:, start:end]
# 
# 
#         #     # Преобразуем DataFrame в длинный формат для Seaborn
#         # df_long = df_segment.reset_index().melt(id_vars="index", var_name="Column", value_name="Value")
# 
#         # print(df_long)
# 
#         # Создаем палитру с красными и синими тонами
#         palette = {0: "red", 1: "blue"}
# 
#         # Строим график
#         df_c_long = df_combined.reset_index().melt(id_vars=["index","dataset"], var_name="Column", value_name="Value")
#         #df_n_long = df_n.reset_index().melt(id_vars="index", var_name="Column", value_name="Value")
#         sns.lineplot(data=df_c_long, x="Column", y="Value",hue='dataset',units="index",
#             estimator=None,palette=palette)
#         #sns.lineplot(data=df_n_long, x="Column", y="Value")
# 
# 
# 
#         #     # Подписи графика
#         # plt.title(f"График {i+1} (столбцы {start}-{end-1})")
#         # plt.xlabel("Столбцы")
#         # plt.ylabel("Значения")
#         # plt.legend(title="Строки")
#             
#         #     # Отображаем график
#         plt.show(block=False)
#         # #plt.show()
# 
#         #----------------------------QDA--QDA--QDA--QDA--QDA--QDA--------------------------
#         print('QDA')
# 
#         X = df_combined.drop(columns=['dataset'])  # Признаки
#         y = df_combined['dataset']
# 
#         # 3. Разделение на обучающую и тестовую выборки
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# 
# 
#         corr_matrix = X_train.corr().abs()  # Вычисляем корреляцию по модулю
#         upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#         self.to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.90)]
# 
#         print(f'to_drop: {self.to_drop}')
# 
# 
#         X_train = X_train.drop(columns=self.to_drop)
#         X_test = X_test.drop(columns=self.to_drop)
# 
#         print(X_train.shape)
#         # ==== 3. Применяем PCA ====
#         # n_components = 80  # Можно варьировать 100-500
#         # pca = PCA(n_components=n_components)
# 
#         # X_train_pca = pca.fit_transform(X_train)  # Обучаем PCA на train
#         # X_test_pca = pca.transform(X_test)  # Применяем к test
# 
#         # ==== 4. Обучаем QDA ====
#         self.qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
#         self.qda.fit(X_train, y_train)
# 
#         # 5. Предсказание и оценка точности
#         y_pred = self.qda.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
# 
#         qda_probs = self.qda.predict_proba(X_test)
#         #print(f'qda_probs: {qda_probs:.2f}')
#         print('qda_probs')
#         print(qda_probs)
#         confidence_scores = np.max(qda_probs, axis=1) 
#         print('confidence_scores')
#         print(confidence_scores)
# 
#         uncertainty_scores = entropy(qda_probs.T)
#         print('uncertainty_scores')
#         print(uncertainty_scores)
# 
# 
#         print(f'Accuracy: {accuracy:.2f}')
#         return self.qda
# 
#     def QDAfit(self,folder_path):
# 
#         df_fd=self.view_data(folder_path)
# 
#         df_fd1=df_fd.copy()
#      
# 
#         df_p1["dataset"]="1"
#         df_n1["dataset"]="0"
# 
#         #print(df_p)
#         #print(df_n)
# 
#         df_combined = pd.concat([df_p1, df_n1])
# 
#         print(df_combined)
# 
# 
# 
#         plt.show(block=False)
# 
# 
#         #----------------------------QDA--QDA--QDA--QDA--QDA--QDA--------------------------
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
# class Some_Processor:
# =============================================================================    
   