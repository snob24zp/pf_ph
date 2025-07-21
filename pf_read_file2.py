import numpy as np
import re
import os
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.stats import entropy

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif





import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import plotly.colors as pc
from plotly.subplots import make_subplots

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
                        #print(values)
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
            #print(data_array[:])
            #print(data_array.shape)

            data_array_1 = data_array.flatten('F')[self.data_points:]

            #print(data_array_1[:])
            #print(data_array_1.shape)
            
            if  self.data is None:
                self.data = np.copy(data_array_1)
            else:
                self.data = np.vstack((self.data, data_array_1))

        #print(self.data[:])
        #print(self.data.shape)
        #print(self.data_headers)
        df = pd.DataFrame(self.data)
        #df.columns=self.data_headers[2:2+self.data_headers_number]

        #print(df)
        accepted_types_p = {'+', 'y', 'pass', 'p'}
        accepted_types_n = {'-', 'n', 'fail', 'f'}

        #print("Result type is accepted.")
        if result_type in accepted_types_p:
            self.df_p=df
        elif result_type in accepted_types_n:
            self.df_n=df
        else: DataFileFormatError("unknown result type")
        return df, files

# используем дополнильные предположения об определённых параметрах    
class Some_Processor:
    def __init__(self):
        self.periods=None
        self.df=None
        self.data_points=None
    def get_params(self,DR):
        self.data_points=DR.data_points
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

    def avg_datatype(self,df=None):
        
        if df is not None:
            self.df=df.copy()
            
        original_columns = list(self.df.columns)
        dataset_index = original_columns.index("dataset")
        
        # Группировка
        df_avg = (
            self.df.groupby("dataset")
              .mean(numeric_only=True)
              .reset_index()
        )
        
        # Перестановка колонок: вернём 'dataset' туда, где он был
        cols = list(df_avg.columns)
        cols.remove("dataset")
        cols.insert(dataset_index, "dataset")
        df_avg = df_avg[cols]
        
        self.df=df_avg
        return self.df
    
    def wide2chunk(self,step=None,df=None):
        
        if df is not None:
            self.df=df.copy()
            
        if step is None:
            step=1
            
        step = self.data_points*step
            

       # step = self.data_points
        rows = []
        
        for dataset_val in [0, 1]:
            df_subset = (self.df[self.df['dataset'] 
                                 == dataset_val].drop(columns='dataset')
                        )
            arr = df_subset.to_numpy()
        
            for series_id, row in enumerate(arr):
                num_chunks = len(row) // step
                for chunk_id in range(num_chunks):
                    chunk = row[chunk_id * step : (chunk_id + 1) * step]
                    for point_id, val in enumerate(chunk):
                        rows.append({
                            'dataset': dataset_val,
                            'series_id': series_id,
                            'chunk_id': chunk_id,
                            'point_id': point_id,
                            'value': val
                        })
        
        df_chunks = pd.DataFrame(rows)
        self.df=df_chunks
        #print('df_chunks',df_chunks)
        return df_chunks
    
    def chunk2wide(self, df=None):
        """
        Преобразует DataFrame из длинного chunk-формата обратно в широкий формат.
        Требуется, чтобы были колонки:
        ['dataset', 'series_id', 'chunk_id', 'point_id', 'value']
        """
    
        if df is not None:
            self.df = df.copy()
    
        # Создадим колонку абсолютной позиции точки
        self.df['abs_point_id'] = (
            self.df['chunk_id'] * self.data_points + self.df['point_id']
        )
    
        # Упорядочим и перегруппируем
        wide_df = (
            self.df.sort_values(['dataset', 'series_id', 'abs_point_id'])
            .pivot_table(
                index=['dataset', 'series_id'],
                columns='abs_point_id',
                values='value'
            )
            .reset_index()
        )
    
        # Приведение типа колонок к int (если нужно)
        wide_df.columns.name = None  # убираем имя колонок
        wide_df.columns = ['dataset', 'series_id'] + sorted(
            [int(c) for c in wide_df.columns[2:]]
        )
        #wide_df.drop(columns='series_id')
        wide_df = wide_df[wide_df.columns[2:].tolist() + wide_df.columns[:1].tolist()]
        self.df = wide_df
        return wide_df
    
    def chunks_centering(self):
        self.df['value'] -= self.df.groupby(
            ['dataset', 'series_id', 'chunk_id'])['value'].transform('mean')
        return self.df

    def half_sum_dif(self,df=None):
        
        if df is not None:
            self.df=df.copy()
        
        step=self.data_points    
        
        # Убедимся, что все названия колонок — строки
        #self.df.columns = self.df.columns.astype(str)
    
        # Преобразуем числовые колонки к int (чтобы избежать ошибок при срезах)
        #num_cols = [int(col) for col in self.df.columns if col != 'dataset']
        #num_cols.sort()
    
        # A: первые 12*шаг колонок
        #A_cols = list(map(str, num_cols[:12 * step]))
        #B_cols = list(map(str, num_cols[12 * step:24 * step]))
    
        # Копии данных
        A = self.df.iloc[:,:12 * step].values
        B = self.df.iloc[:,12 * step:24 * step].values
    
        # Вычисляем новые значения
        self.df.iloc[:,:12 * step] = (A - B)/2
        self.df.iloc[:,12 * step:24 * step] = (A + B)/2
    
        return self.df


    def subtract_base_chunk(self,df=None):
        """
        Вычитает из 'value' в каждой группе (dataset, series_id, chunk_id)
        среднее значение по 8 наименьшим point_id в этой группе.
        """
        if df is not None:
            self.df=df.copy()
        group_keys = ['dataset', 'series_id', 'chunk_id']
        
        #print(self.param_dict)
        
        base_size=int(self.param_dict['Baseline']*
                      self.param_dict['Acquired data point per second'])
    
        def process_group(group):
            # Среднее по 8 минимальным point_id
            #print(group)
            baseline = group.nsmallest(base_size, 'point_id')['value'].mean()
            # Вычитание baseline из всей группы
            group['value'] = group['value'] - baseline
            return group
    
        # Применение ко всем группам
        self.df = self.df.groupby(group_keys, group_keys=False).apply(process_group)
        return self.df 
    
    def del_peaks(self):
        df_filtered = self.df.mask(self.df.abs() > 1000)
        
        # Применим линейную интерполяцию по строкам
        self.df = df_filtered.interpolate(axis=1, method='linear', limit_direction='both')
        
        return self.df  

class Statistical_Processor:
    def __init__(self):
        self.pca_importance=None
        self.mi_importance=None
    def pca_analize(self,SP):
        X = SP.df.copy()
        #print('x',X)
        X.columns = X.columns.astype(str)
        
        #X.to_csv("data.csv", index=False, sep=",", encoding="utf-8")
        
        target_names = 'dataset'
        
        y = X['dataset']
        X = X.drop(columns=['dataset'])
        feature_names = X.columns.astype(str)
        
        #print('x',X)
        # Шаг 2: Стандартизируем данные
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Шаг 3: PCA
        pca = PCA(5)
        X_pca = pca.fit_transform(X_scaled)
        
        # Получим веса признаков для каждой компоненты
        components = pca.components_  # shape: (n_components, n_features)
        explained = pca.explained_variance_ratio_  # shape: (n_components,)
        
        # Квадраты компонент (вкладов) — аналог абсолютной "важности"
        squared_loadings = components ** 2  # shape: (n_components, n_features)
        
        # Взвешиваем их на объяснённую дисперсию
        weighted_importance = np.dot(explained, squared_loadings)  # shape: (n_features,)
        
        # Соберём в датафрейм
        importance_df = pd.DataFrame({
            'feature': X.columns.astype(int),
            'importance': weighted_importance
        }).sort_values('importance', ascending=False)        
        self.pca_importance=importance_df
        


# =============================================================================
#         # === График 1: 2D-проекция данных ===
#         plt.figure(figsize=(8, 6))
#         scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Set1', edgecolor='k')
#         plt.xlabel("PC1")
#         plt.ylabel("PC2")
#         plt.title("PCA Projection")
#         plt.legend(handles=scatter.legend_elements()[0], labels=["0", "1"], title="Class")
#         plt.show()
#         
# =============================================================================
# =============================================================================
#         # === График 2: Scree Plot ===
#         plt.figure(figsize=(8, 6))
#         plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
#                  pca.explained_variance_ratio_, marker='o')
#         plt.xlabel("Component")
#         plt.ylabel("Explained Variance Ratio")
#         plt.title("Scree Plot")
#         plt.grid(True)
#         plt.show()
#         
# =============================================================================
# =============================================================================
#         # === График 3: Накопленная дисперсия ===
#         plt.figure(figsize=(8, 6))
#         plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
#         plt.xlabel("Number of Components")
#         plt.ylabel("Cumulative Explained Variance")
#         plt.title("Cumulative Variance")
#         plt.grid(True)
#         plt.show()
#         
# =============================================================================
# =============================================================================
#         # === График 4: Тепловая карта компонент ===
#         plt.figure(figsize=(12, 6))
#         sns.heatmap(pca.components_, cmap="viridis",
#                     xticklabels=feature_names,
#                     yticklabels=[f"PC{i+1}" for i in range(pca.n_components_)])
#         plt.title("PCA Components (Feature Weights)")
#         plt.xlabel("Features")
#         plt.ylabel("Principal Components")
#         plt.show()
#         
# =============================================================================
# Выберем 10 признаков с наибольшей важностью (тип int)
        top_features = importance_df.sort_values(by="importance", ascending=False).head(300)
        
        # Получаем список признаков и их индексы
        top_feature_ids = top_features["feature"].values  # это int
        top_feature_indices = [X.columns.get_loc(str(f)) for f in top_feature_ids]  # позиции признаков в X
        
        # Biplot
        plt.figure(figsize=(20, 10))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Set1', alpha=0.6, edgecolor='k')
        
        # Рисуем стрелки и подписи только для топ-10 признаков
        for idx in top_feature_indices:
            plt.arrow(0, 0, pca.components_[0, idx] * 1000, pca.components_[1, idx] * 1000,
                      color='blue', alpha=0.5, head_width=0.1)
            plt.text(pca.components_[0, idx] * 1100, pca.components_[1, idx] * 1100,
                     str(X.columns[idx]),  # подпись — номер признака
                     color='blue', ha='center', va='center')
        
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Biplot (Top 10 Features)")
        plt.grid(True)
        plt.show()        
# =============================================================================
#         # === График 6: Проекция с классами ===
#         plt.figure(figsize=(8, 6))
#         sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set1', legend="full")
#         plt.xlabel("PC1")
#         plt.ylabel("PC2")
#         plt.title("Cluster Map in PCA Space")
#         plt.show()
#         
# =============================================================================
        return importance_df
    
    def mi_analize(self,SP):
        X = SP.df.copy()
        #print('x',X)
        X.columns = X.columns.astype(str)
        
        #X.to_csv("data.csv", index=False, sep=",", encoding="utf-8")
        
        target_names = 'dataset'
        
        y = X['dataset']
        X = X.drop(columns=['dataset'])
        
        # Стандартизация
        X_scaled = StandardScaler().fit_transform(X)
        
        # Вычисление взаимной информации
        mi_scores = mutual_info_classif(X_scaled, y, discrete_features=False, random_state=0)
        
        # Сборка DataFrame
        mi_df = pd.DataFrame({
            'feature': X.columns.astype(str),
            'mutual_info': mi_scores
        }).sort_values(by="mutual_info", ascending=False)
        
        # Сортировка по убыванию взаимной информации
        mi_sorted = mi_df.sort_values(by="mutual_info", ascending=False).reset_index(drop=True)
        
        # Добавим колонку "rank" — индекс в отсортированном списке
        mi_sorted["rank"] = mi_sorted.index + 1  # начинаем с 1
        
        # Построение графика
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=mi_sorted, x="rank", y="mutual_info", marker="o", color='blue')
        plt.xlabel("Number of Features (Rank)")
        plt.ylabel("Mutual Information")
        plt.title("Mutual Information by Feature Rank")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        
        # 2. График: Распределение MI
        plt.figure(figsize=(8, 6))
        sns.histplot(mi_df["mutual_info"], bins=30, kde=True)
        plt.title("Distribution of Mutual Information Scores")
        plt.xlabel("Mutual Information")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        
        # 3. График: Корреляция топ‑20 по MI
        top_20 = mi_sorted.head(20)["feature"] 
        plt.figure(figsize=(12, 10))
        sns.heatmap(X[top_20].corr(), cmap="coolwarm", square=True)
        plt.title("Correlation Matrix of Top 20 MI Features")
        plt.tight_layout()
        plt.show()
        
        # Вывод топ-10 признаков по взаимной информации
        #print(mi_df.head(10))
        
        mi_importance_df = pd.DataFrame({
                'feature': X.columns.astype(int),
                'importance': mi_scores
            }).sort_values('importance', ascending=False)
        
        self.mi_importance=mi_importance_df
        return self.mi_importance
    
    def class_balance(self,SP):
        self.part1=SP.df["dataset"].mean()
        print("class_balance ",self.part1)
        

class Individual_Processor:
    def __init__(self):
        self.periods=None
        self.df=None
        self.data_points=None
        self.param_dict={}
    def get_params(self,SP):
        self.data_points=SP.data_points
        self.param_dict=SP.param_dict.copy()
        self.periods = SP.periods
        self.df=SP.df.copy()

    def subtract_base_chunk(self,df=None):
        """
        Вычитает из 'value' в каждой группе (dataset, series_id, chunk_id)
        среднее значение по 8 наименьшим point_id в этой группе.
        """
        if df is not None:
            self.df=df.copy()
        group_keys = ['dataset', 'series_id', 'chunk_id']
        
        #print(self.param_dict)
        
        base_size=int(self.param_dict['Baseline']*
                      self.param_dict['Acquired data point per second'])
    
        def process_group(group):
            # Среднее по 8 минимальным point_id
            #print(group)
            baseline = group.nsmallest(base_size, 'point_id')['value'].mean()
            # Вычитание baseline из всей группы
            group['value'] = group['value'] - baseline
            return group
    
        # Применение ко всем группам
        self.df = self.df.groupby(group_keys, group_keys=False).apply(process_group)
        return self.df    
        
    def base_pause(self):
        pass
    def fourie(self):
        pass
    def exp_raise_fail(self):
        pass
        

class Data_Show2:
    def Data_show(self, SP, fig_name):
        
# =============================================================================
#         if len(SP.df.columns)==5:
#             df=
# 
# =============================================================================
        df_c_long = SP.df.reset_index().melt(
            id_vars=["index", "dataset"],
            var_name="Column",
            value_name="Value"
        )

        # Convert column to string (for Plotly compatibility)
        df_c_long["Column"] = df_c_long["Column"].astype(str)

        # Convert dataset to string for proper coloring
        df_c_long["dataset"] = df_c_long["dataset"].astype(str)

        fig = px.line(
            df_c_long,
            x="Column",
            y="Value",
            color="dataset",
            line_group="index",
            title=fig_name,
            color_discrete_map={"0": 'red', "1": 'blue'},
            category_orders={"dataset": ["0", "1"]}
        )

        fig.update_layout(
            hovermode='closest',
            xaxis_title='Features',
            yaxis_title='Values',
            height=600,
            width=1000
        )

        # Line for phases
        periods = SP.periods

        # Convert to number of data points
        segment_lengths = [(name, int(round(seconds * SP.param_dict['Acquired data point per second'])))
                           for name, seconds in periods]
        segment_lengths[-1] = (segment_lengths[-1][0], segment_lengths[-1][1] + 1)

        # Calculate boundary positions
        boundaries = []
        current_pos = 0
        for sensor in range(24):
            for name, length in segment_lengths:
                boundaries.append((current_pos, name + ' ' + str(sensor)))
                current_pos += length

        # Map point positions to columns
        vlines = []
        for pos, label in boundaries:
            col_name = str(pos)
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
            margin=dict(t=80)  # Slightly increase top margin
        )

        # Show the plot in browser
        fig.write_html(fig_name + ".html")
        webbrowser.open(fig_name + ".html")

    def Data_show_st(self, SP, fig_name, statistic):

        df_c_long = SP.df.reset_index().melt(
            id_vars=["index", "dataset"],
            var_name="Column",
            value_name="Value"
        )

        df_c_long["Column"] = df_c_long["Column"].astype(str)

        # Line for phases
        periods = SP.periods
        segment_lengths = [(name, int(round(seconds * SP.param_dict['Acquired data point per second'])))
                           for name, seconds in periods]
        segment_lengths[-1] = (segment_lengths[-1][0], segment_lengths[-1][1] + 1)

        boundaries = []
        current_pos = 0
        for sensor in range(24):
            for name, length in segment_lengths:
                boundaries.append((current_pos, name + ' ' + str(sensor)))
                current_pos += length

        vlines = []
        for pos, label in boundaries:
            col_name = str(pos)
            vlines.append((col_name, label))

        shapes = []
        annotations = []

        for col, label in vlines:
            shapes.append(dict(
                type="line",
                x0=col, x1=col,
                y0=0, y1=1,
                xref="x", yref="y domain",
                line=dict(color="black", width=1, dash="dot")
            ))
            annotations.append(dict(
                x=col,
                y=1.0,
                xref="x",
                yref="y domain",
                text=label,
                showarrow=False,
                font=dict(size=10, color="black"),
                xanchor="left",
                yanchor="bottom",
                textangle=-90
            ))

        pca_importance_df = statistic.pca_importance.copy()
        mi_importance_df = statistic.mi_importance.copy()

        pca_importance_df['feature'] = pca_importance_df['feature'].astype(str)
        mi_importance_df['feature'] = mi_importance_df['feature'].astype(str)

        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.8, 0.1, 0.1],
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(fig_name, "PCA Importance", "MI Importance")
        )
        fig.update_yaxes(domain=[0.3, 0.85], row=1, col=1)

        dfg = df_c_long.groupby(["dataset", "index"])
        for key, group in dfg:
            fig.add_trace(
                go.Scatter(x=group["Column"], y=group["Value"],
                           mode='lines',
                           name=f"dataset {key[0]} - idx {key[1]}",
                           line=dict(color='red' if key[0] == 0 else 'blue')),
                row=1, col=1
            )

        fig.update_layout(
            shapes=shapes,
            annotations=annotations,
            margin=dict(t=320)
        )

        fig.add_trace(
            go.Heatmap(
                z=[pca_importance_df['importance'].values],
                x=pca_importance_df['feature'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='PCA Importance', y=0.2, len=0.2)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Heatmap(
                z=[mi_importance_df['importance'].values],
                x=mi_importance_df['feature'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='MI Importance', y=0, len=0.2)
            ),
            row=3, col=1
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            title=fig_name,
            margin=dict(t=50, b=50),
        )

        fig.write_html(fig_name + ".html")
        webbrowser.open(fig_name + ".html")


    def Data_show_chunks(self, SP, fig_name):
        df_chunks = SP.df
        palette_25_blues = sample_colorscale(pc.sequential.Blues, np.linspace(0.5, 1.0, 25))
        palette_25_reds = sample_colorscale(pc.sequential.Reds, np.linspace(0.5, 1.0, 25))

        df_0 = df_chunks[df_chunks['dataset'] == 0]
        df_1 = df_chunks[df_chunks['dataset'] == 1]

        fig = go.Figure()

        chunk_ids_0 = df_0['chunk_id'].unique()
        for i, chunk_id in enumerate(chunk_ids_0):
            chunk_df = df_0[df_0['chunk_id'] == chunk_id]
            pd.set_option('display.max_rows', None)  # Показывать все строки
            #print(chunk_df['point_id'])
            #print(len(chunk_df['point_id']))
            
            x_vals = []
            y_vals = []
            
            points_per_period = SP.data_points  # обычно 121
            num_points = len(chunk_df)
            num_periods = num_points // points_per_period
            
            for p in range(num_periods):
                chunk = chunk_df.iloc[p * points_per_period : (p + 1) * points_per_period]
                x_vals.extend(chunk['point_id'].tolist())
                y_vals.extend(chunk['value'].tolist())
                
                # Добавим разрыв между периодами
                x_vals.append(None)
                y_vals.append(None)
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name=f'ds=0, ch={chunk_id}',
                line=dict(color=palette_25_blues[i % len(palette_25_blues)]),
                #legendgroup='0'
            ))

        chunk_ids_1 = df_1['chunk_id'].unique()
        for i, chunk_id in enumerate(chunk_ids_1):
            chunk_df = df_1[df_1['chunk_id'] == chunk_id]
            
            x_vals = []
            y_vals = []
            
            points_per_period = SP.data_points  # обычно 121
            num_points = len(chunk_df)
            num_periods = num_points // points_per_period
            
            for p in range(num_periods):
                chunk = chunk_df.iloc[p * points_per_period : (p + 1) * points_per_period]
                x_vals.extend(chunk['point_id'].tolist())
                y_vals.extend(chunk['value'].tolist())
                
                # Добавим разрыв между периодами
                x_vals.append(None)
                y_vals.append(None)
            
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name=f'ds=1, ch={chunk_id}',
                line=dict(color=palette_25_reds[i % len(palette_25_reds)]),
                #legendgroup='1'
            ))

        fig.update_layout(
            title="Chunk-wise Plots (121 points each, grouped by dataset)",
            xaxis_title="point_id",
            yaxis_title="value",
            height=600,
            width=1000
        )

        periods = SP.periods
        segment_lengths = [(name, int(round(seconds * SP.param_dict['Acquired data point per second'])))
                           for name, seconds in periods]
        segment_lengths[-1] = (segment_lengths[-1][0], segment_lengths[-1][1] + 1)

        boundaries = []
        current_pos = 0
        ux = len(chunk_df['point_id'].unique())
        sensors = ux // SP.data_points
        for sensor in range(sensors):
            for name, length in segment_lengths:
                boundaries.append((current_pos, name + ' ' + str(sensor)))
                current_pos += length

        vlines = []
        for pos, label in boundaries:
            col_name = str(pos)
            vlines.append((col_name, label))

        shapes = []
        annotations = []

        for col, label in vlines:
            shapes.append(dict(
                type="line",
                x0=int(col), x1=int(col),
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="black", width=1, dash="dot")
            ))
            annotations.append(dict(
                x=int(col),
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
            margin=dict(t=80)
        )


        fig.write_html(fig_name + ".html")
        webbrowser.open(fig_name + ".html")
        
    



class ProccesingFFE:
    def __init__(self):
        self.DR=None
        self.DSh=None
        self.SP=None
        self.SP12=None
    def view(self,folder_pass_path,folder_fail_path):
        self.DR=DataRead()
        self.DSh=Data_Show2()
        self.SP=Some_Processor()
        self.DR.read_result(folder_pass_path,'+')
        self.DR.read_result(folder_fail_path,'-')
        self.SP.get_params(self.DR)
        self.SP.combo_result(self.DR.df_p,self.DR.df_n)
        #self.SP.del_pause("Pause")
        #self.DSh.Data_show(self.SP,"each_sensor")
        self.SP.avg_datatype()
        #self.DSh.Data_show(self.SP,"avg_sensor")
        self.SP12=copy.deepcopy(self.SP)
        self.SP.wide2chunk()
        self.DSh.Data_show_chunks(self.SP,"avg_sensor_separate")
        self.SP12.half_sum_dif()
        self.DSh.Data_show(self.SP12,"half_sum_dif")
        self.SP12.wide2chunk(step=12)
        self.SP12.chunks_centering()
        self.DSh.Data_show_chunks(self.SP12,"half_sum_dif_2chunks")
    
    def eda(self,folder_pass_path,folder_fail_path):
        self.DR=DataRead()
        self.DSh=Data_Show2()
        self.SP=Some_Processor()
        self.StP=Statistical_Processor()
        self.DR.read_result(folder_pass_path,'+')
        self.DR.read_result(folder_fail_path,'-')
        self.SP.get_params(self.DR)
        self.SP.combo_result(self.DR.df_p,self.DR.df_n)
        #self.SP.del_pause("Pause")
        #self.DSh.Data_show(self.SP,"each_sensor")
        #self.SP.avg_datatype()
        #self.DSh.Data_show(self.SP,"avg_sensor")
        #self.SP12=copy.deepcopy(self.SP)
        self.SP.half_sum_dif()
        self.DSh.Data_show(self.SP,"each_halfsumdif_sensor")
        self.StP.pca_analize(self.SP)
        #self.DSh.Data_show_st(self.SP,"each_halfsumdif_sensor",statistical=self.StP)
        self.StP.mi_analize(self.SP)
        self.DSh.Data_show_st(self.SP,"each_halfsumdif_sensor",statistic=self.StP)
        
    def fe(self,folder_pass_path,folder_fail_path):      
        self.DR=DataRead()
        self.DSh=Data_Show2()
        self.SP=Some_Processor()
        self.StP=Statistical_Processor()
        #self.IP=Individual_Processor()
        
        self.DR.read_result(folder_pass_path,'+')
        self.DR.read_result(folder_fail_path,'-')
        self.SP.get_params(self.DR)
        self.SP.combo_result(self.DR.df_p,self.DR.df_n)
        self.StP.class_balance(self.SP)
        #self.DSh.Data_show(self.SP,"each_sensor, fe")
        self.SP.del_peaks()
        #self.DSh.Data_show(self.SP,"each_sensor_interpolation, fe")
        self.SP.half_sum_dif()
        #self.SP.avg_datatype()
        
        self.SP.wide2chunk()
        self.SP.subtract_base_chunk()
        #self.DSh.Data_show_chunks(self.SP,"sensor_separate_sub_base")
        self.SP.chunk2wide()
        
        self.DSh.Data_show(self.SP,"each_sensor_wo_base")

        
#Chicken Data Combined FAIL\Chicken Data Combined FAIL

      
if __name__=='__main__':
    Pr=ProccesingFFE()
    #folder_pass_path = "./Chicken Data Combined PASS/Chicken Data Combined PASS"
    #folder_fail_path = "./Chicken Data Combined FAIL/Chicken Data Combined FAIL"
    #folder_pass_path = "./p3/p3"
    #folder_fail_path = "./n3/n3"
    folder_pass_path = "./p2/p2"
    folder_fail_path = "./n2/n2"

    #Pr.view(folder_pass_path,folder_fail_path)
    Pr.eda(folder_pass_path, folder_fail_path)
    #Pr.fe(folder_pass_path, folder_fail_path)

