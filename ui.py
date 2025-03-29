import pf_read_file
import tkinter as tk
from tkinter import filedialog

import pf_read_file

class ModelTrainerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Trainer")
        self.root.geometry("500x400")
        self.qda=None
        
        # Поля выбора каталога
        self.create_directory_selector("Catalog with Pass files", 0, "pass_files_dir")
        self.create_directory_selector("Catalog with Fial files", 1, "fail_files_dir")
        self.create_directory_selector("Model", 2, "model_dir")
        self.create_directory_selector("Catalog with files for analizis", 3, "analysis_files_dir")
        ModelTrainerUI.Nrow=1
        # Поле текущая модель
        tk.Label(root, text="Current Model:").grid(row=3+ModelTrainerUI.Nrow, column=0, padx=10, pady=5, sticky="w")
        self.model_var = tk.StringVar()
        tk.Entry(root, textvariable=self.model_var, state='readonly').grid(row=3+ModelTrainerUI.Nrow, column=1, padx=10, pady=5)
        
        # Кнопки управления
        tk.Button(root, text="Train Model", command=self.train_model).grid(row=4+ModelTrainerUI.Nrow, column=0, padx=10, pady=5)
        tk.Button(root, text="Save Model", command=self.save_model).grid(row=4+ModelTrainerUI.Nrow, column=1, padx=10, pady=5)
        tk.Button(root, text="Load Model", command=self.load_model).grid(row=4+ModelTrainerUI.Nrow, column=2, padx=10, pady=5)
        tk.Button(root, text="Analyze Files", command=self.analyze_files).grid(row=4+ModelTrainerUI.Nrow, column=3, padx=10, pady=5)
        
        # Поле списка
        tk.Label(root, text="List:").grid(row=5+ModelTrainerUI.Nrow, column=0, padx=10, pady=5, sticky="w")
        self.listbox = tk.Listbox(root, height=5, width=50)
        self.listbox.grid(row=6+ModelTrainerUI.Nrow, column=0, columnspan=3, padx=10, pady=5)

    def create_directory_selector(self, label_text, row, var_name):
        tk.Label(self.root, text=label_text).grid(row=row, column=0, padx=10, pady=5, sticky="w")
        entry = tk.Entry(self.root, width=40)
        entry.grid(row=row, column=1, padx=10, pady=5)
        tk.Button(self.root, text="Browse", command=lambda: self.select_directory(entry, var_name)).grid(row=row, column=2, padx=10, pady=5)
        setattr(self, var_name, "")  # Инициализация переменной для хранения пути

    def select_directory(self, entry, var_name):
        directory = filedialog.askdirectory()
        if directory:
            entry.delete(0, tk.END)
            entry.insert(0, directory)
            setattr(self, var_name, directory)  # Сохранение пути в атрибут класса
    
    def train_model(self):
        print("Training the model...")
        self.model_var.set("Model Trained")
        self.qda=pf_read_file.Qda()
        self.qda.QDAanalysis(self.pass_files_dir,self.fail_files_dir)
    
    def save_model(self):
        print("Saving the model...")
    
    def load_model(self):
        print("Loading the model...")
        self.model_var.set("Loaded Model")
    
    def analyze_files(self):
        if not self.analysis_files_dir:
            print("Please select a directory for analysis files.")
            return
        print(f"Analyzing files in: {self.analysis_files_dir}")
        # Здесь можно добавить вызов метода анализа файлов из pf_read_file
        if self.qda:
            fa=self.qda.classify_files(self.analysis_files_dir)
        else:
            print("QDA model is not initialized. Train the model first.")
        for file, label in fa.items():
            self.listbox.insert(tk.END, f"{file}: {label}") 
        


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelTrainerUI(root)
    root.mainloop()

