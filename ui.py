import pf_read_file
import tkinter as tk
from tkinter import filedialog

class ModelTrainerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Trainer")
        self.root.geometry("500x400")
        
        # Поля выбора каталога
        self.create_directory_selector("Catalog with Pass files", 0)
        self.create_directory_selector("Catalog with Fial files", 1)
        self.create_directory_selector("Model", 2)
        self.create_directory_selector("Catalog with files for analizis", 3)
        ModelTrainerUI.Nrow=1
        # Поле текущая модель
        tk.Label(root, text="Current Model:").grid(row=3+ModelTrainerUI.Nrow, column=0, padx=10, pady=5, sticky="w")
        self.model_var = tk.StringVar()
        tk.Entry(root, textvariable=self.model_var, state='readonly').grid(row=3+ModelTrainerUI.Nrow, column=1, padx=10, pady=5)
        
        # Кнопки управления
        tk.Button(root, text="Train Model", command=self.train_model).grid(row=4+ModelTrainerUI.Nrow, column=0, padx=10, pady=5)
        tk.Button(root, text="Save Model", command=self.save_model).grid(row=4+ModelTrainerUI.Nrow, column=1, padx=10, pady=5)
        tk.Button(root, text="Load Model", command=self.load_model).grid(row=4+ModelTrainerUI.Nrow, column=2, padx=10, pady=5)
        
        # Поле списка
        tk.Label(root, text="List:").grid(row=5+ModelTrainerUI.Nrow, column=0, padx=10, pady=5, sticky="w")
        self.listbox = tk.Listbox(root, height=5, width=50)
        self.listbox.grid(row=6+ModelTrainerUI.Nrow, column=0, columnspan=3, padx=10, pady=5)

    def create_directory_selector(self, label_text, row):
        tk.Label(self.root, text=label_text).grid(row=row, column=0, padx=10, pady=5, sticky="w")
        entry = tk.Entry(self.root, width=40)
        entry.grid(row=row, column=1, padx=10, pady=5)
        tk.Button(self.root, text="Browse", command=lambda: self.select_directory(entry)).grid(row=row, column=2, padx=10, pady=5)

    def select_directory(self, entry):
        directory = filedialog.askdirectory()
        if directory:
            entry.delete(0, tk.END)
            entry.insert(0, directory)
    
    def train_model(self):
        print("Training the model...")
        self.model_var.set("Model Trained")
    
    def save_model(self):
        print("Saving the model...")
    
    def load_model(self):
        print("Loading the model...")
        self.model_var.set("Loaded Model")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelTrainerUI(root)
    root.mainloop()

