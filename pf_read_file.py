import numpy as np
import re
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

folder_path = "./p"

# Получаем список всех файлов в папке
files = os.listdir(folder_path)

data= np.empty((0,121*25), float)  # Создаем пустой 2D массив с 3 столбцами

# Добавляем строки


# Проходимся по файлам
for file in files:
    file_path = os.path.join(folder_path, file)  # Полный путь к файлу
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

# Строим матрицу корреляций
selected_cols = [0, 1*121, 2*121,24*121]  
corr_matrix = df[selected_cols].corr()


# Визуализируем
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Матрица корреляций")
plt.show()



df_selected = df.iloc[:, 0:121]
# Преобразуем DataFrame в длинный формат (чтобы Seaborn понял)
df_selected_long = df_selected.reset_index().melt(id_vars="index", var_name="Column", value_name="Value")

# Строим график с Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_selected_long, x="Column", y="Value", hue="index")

plt.title("Графики строк DataFrame (Seaborn, столбцы 0-120)")
plt.xlabel("Столбцы")
plt.ylabel("Значения")
plt.legend(title="Строки")
plt.show()