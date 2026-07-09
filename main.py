# main.py
# !/usr/bin/env python3
"""
Минеральный Анализатор - Объединенный РФА и Раман анализ
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Добавление текущей директории в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui_main import MineralAnalyzerApp


def main():
    """Точка входа в приложение"""
    root = tk.Tk()

    # Установка иконки
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
    if os.path.exists(icon_path):
        try:
            root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Не удалось загрузить иконку: {e}")

    # Установка стиля
    style = ttk.Style()
    style.theme_use('clam')

    app = MineralAnalyzerApp(root)

    # Обработка закрытия
    def on_closing():
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()