import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# === ДАННЫЕ ===
# Генерируем фиктивные данные для двух классов
np.random.seed(42)
X_train = np.vstack([
    np.random.normal(loc=0, scale=1, size=(50, 3)),  # Класс 0
    np.random.normal(loc=2, scale=1, size=(50, 3))   # Класс 1
])
y_train = np.hstack([
    np.zeros(50),  # Метки для класса 0
    np.ones(50)    # Метки для класса 1
])

# === ОБУЧЕНИЕ QDA ===
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# === ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ ===
def predict_qda():
    try:
        # Получаем введенные пользователем значения
        values = [float(entry.get()) for entry in entry_fields]
        X_new = np.array(values).reshape(1, -1)  # Преобразуем в массив
        predicted_class = qda.predict(X_new)[0]  # Делаем предсказание
        
        # Выводим результат
        messagebox.showinfo("Результат", f"Предсказанный класс: {int(predicted_class)}")
    except ValueError:
        messagebox.showerror("Ошибка", "Введите числовые значения!")

# === СОЗДАНИЕ GUI ===
root = tk.Tk()
root.title("QDA Classifier")

# Инструкции
tk.Label(root, text="Введите значения признаков:").pack(pady=5)

# Поля ввода (по количеству признаков)
entry_fields = []
for i in range(3):  # 3 признака
    frame = tk.Frame(root)
    frame.pack(pady=2)
    tk.Label(frame, text=f"Признак {i+1}:").pack(side=tk.LEFT, padx=5)
    entry = tk.Entry(frame, width=10)
    entry.pack(side=tk.RIGHT)
    entry_fields.append(entry)

# Кнопка предсказания
tk.Button(root, text="Предсказать", command=predict_qda).pack(pady=10)

# Запуск интерфейса
root.mainloop()
