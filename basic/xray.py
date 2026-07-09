import re
import zipfile
import os
import tempfile
from collections import defaultdict

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# -------------------------------------------------------------------
# 1. Парсинг файла x.txt – извлечение элементного состава (PPM)
# -------------------------------------------------------------------
def parse_elements_from_file(filepath):
    """
    Возвращает словарь {элемент: ppm} из таблицы после строки 'Эл.         PPM'
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if 'Эл.' in line and 'PPM' in line:
            start = i + 1
            break
    if start is None:
        return {}

    elements = {}
    for line in lines[start:]:
        line = line.strip()
        if not line or line.startswith('<<'):
            break
        parts = line.split()
        if len(parts) >= 3:
            elem = parts[0]
            ppm_str = parts[1].replace(',', '.')
            try:
                ppm = float(ppm_str)
            except ValueError:
                continue
            if re.match(r'^[A-Z][a-z]?$', elem):
                elements[elem] = ppm
    return elements

# -------------------------------------------------------------------
# 2. Парсинг блоков CPS и CPS Light
# -------------------------------------------------------------------
def parse_cps_blocks(filepath):
    """
    Возвращает два списка: cps_data, cps_light_data (чисел с плавающей точкой)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Ищем блок CPS
    cps_pattern = r'<<↓↓CPS↓↓>>\s*(.*?)\s*<<↑↑CPS↑↑>>'
    cps_match = re.search(cps_pattern, content, re.DOTALL)
    cps_data = []
    if cps_match:
        block = cps_match.group(1).strip()
        for token in block.split():
            try:
                cps_data.append(float(token.replace(',', '.')))
            except ValueError:
                pass

    # Ищем блок CPS Light
    cps_light_pattern = r'<<↓↓CPS Light↓↓>>\s*(.*?)\s*<<↑↑CPS Light↑↑>>'
    cps_light_match = re.search(cps_light_pattern, content, re.DOTALL)
    cps_light_data = []
    if cps_light_match:
        block = cps_light_match.group(1).strip()
        for token in block.split():
            try:
                cps_light_data.append(float(token.replace(',', '.')))
            except ValueError:
                pass

    return cps_data, cps_light_data

# -------------------------------------------------------------------
# 3. Парсинг химической формулы (упрощённый)
# -------------------------------------------------------------------
def parse_formula(formula_str):
    pattern = re.compile(r'([A-Z][a-z]?)([0-9]*)')
    matches = pattern.findall(formula_str)
    comp = {}
    for elem, num in matches:
        if elem == 'O':
            continue
        count = int(num) if num else 1
        comp[elem] = comp.get(elem, 0) + count
    return comp

# -------------------------------------------------------------------
# 4. Загрузка базы RRUFF из ZIP-архивов
# -------------------------------------------------------------------
def load_rruff_database(base_dir):
    mineral_db = []   # список {'name': ..., 'formula': ..., 'comp': {...}}

    if not os.path.exists(base_dir):
        return mineral_db

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if not file.endswith('.zip'):
                continue
            zip_path = os.path.join(root, file)
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(tmpdir)

                    for sub_root, sub_dirs, sub_files in os.walk(tmpdir):
                        for f in sub_files:
                            if not f.endswith('.txt'):
                                continue
                            txt_path = os.path.join(sub_root, f)
                            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as tf:
                                content = tf.read()

                            formula = None
                            for line in content.splitlines():
                                if re.search(r'(Formula|FORMULA|Химическая формула)\s*[:=]', line):
                                    parts = re.split(r'[:=]', line, 1)
                                    if len(parts) > 1:
                                        formula = parts[1].strip()
                                        break

                            if formula is None:
                                first_line = content.splitlines()[0].strip()
                                if re.match(r'^[A-Z][a-z]?[0-9]*(?:\s*[A-Z][a-z]?[0-9]*)*$', first_line):
                                    formula = first_line

                            if formula:
                                comp = parse_formula(formula)
                                if comp:
                                    mineral_name = os.path.splitext(file)[0]
                                    mineral_db.append({
                                        'name': mineral_name,
                                        'formula': formula,
                                        'comp': comp
                                    })
                            break
            except Exception as e:
                # Пропускаем проблемные архивы
                continue

    return mineral_db

# -------------------------------------------------------------------
# 5. Сравнение составов (евклидово расстояние)
# -------------------------------------------------------------------
def compare_composition(sample_comp, mineral_comp):
    atomic_masses = {
        'Mg': 24.305, 'K': 39.0983, 'Ti': 47.867, 'Si': 28.0855,
        'Ca': 40.078, 'Nb': 92.906, 'P': 30.9738, 'Ni': 58.6934,
        'Fe': 55.845, 'Mo': 95.95, 'Cr': 51.9961, 'Al': 26.9815,
        'Mn': 54.938, 'Na': 22.99, 'Zn': 65.38, 'Cu': 63.546,
        'Co': 58.933, 'V': 50.9415, 'Ba': 137.327, 'Sr': 87.62,
        'Zr': 91.224, 'Hf': 178.49, 'Ta': 180.948, 'W': 183.84,
        'Pb': 207.2, 'Bi': 208.98, 'Th': 232.038, 'U': 238.029
    }

    sample_moles = {}
    for elem, ppm in sample_comp.items():
        if elem in atomic_masses:
            sample_moles[elem] = ppm / atomic_masses[elem]
    total_sample = sum(sample_moles.values())
    if total_sample == 0:
        return float('inf')
    sample_norm = {e: v / total_sample for e, v in sample_moles.items()}

    total_mineral = sum(mineral_comp.values())
    if total_mineral == 0:
        return float('inf')
    mineral_norm = {e: v / total_mineral for e, v in mineral_comp.items()}

    common = set(sample_norm.keys()) & set(mineral_norm.keys())
    if not common:
        return float('inf')

    diff = 0.0
    for elem in common:
        diff += (sample_norm[elem] - mineral_norm[elem]) ** 2
    for elem in set(sample_norm.keys()) - set(mineral_norm.keys()):
        diff += sample_norm[elem] ** 2
    for elem in set(mineral_norm.keys()) - set(sample_norm.keys()):
        diff += mineral_norm[elem] ** 2

    return diff

# -------------------------------------------------------------------
# 6. Основной класс приложения
# -------------------------------------------------------------------
class RFAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ РФА спектра")
        self.root.geometry("1000x700")

        # Загрузка базы (один раз)
        self.mineral_db = []
        self.status_var = tk.StringVar()
        self.status_var.set("Загрузка базы RRUFF...")
        self.root.update()
        base_dir = 'raw_rruff/xray'
        if os.path.exists(base_dir):
            self.mineral_db = load_rruff_database(base_dir)
            self.status_var.set(f"База загружена: {len(self.mineral_db)} минералов")
        else:
            self.status_var.set("База не найдена (папка raw_rruff/xray отсутствует)")

        # Верхняя панель
        top_frame = tk.Frame(root)
        top_frame.pack(pady=5, fill=tk.X)

        btn_open = tk.Button(top_frame, text="Открыть файл", command=self.open_file)
        btn_open.pack(side=tk.LEFT, padx=5)

        self.file_label = tk.Label(top_frame, text="Файл не выбран", anchor='w')
        self.file_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        self.status_label = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Главная область – график + таблица
        main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Левая часть – график
        left_frame = tk.Frame(main_pane)
        main_pane.add(left_frame, width=700)

        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Данные CPS")
        self.ax.set_xlabel("Номер канала")
        self.ax.set_ylabel("Интенсивность")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Правая часть – таблица элементов и результат
        right_frame = tk.Frame(main_pane, width=300)
        main_pane.add(right_frame)

        # Таблица элементов
        tree_frame = tk.LabelFrame(right_frame, text="Элементный состав (PPM)", padx=5, pady=5)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.tree = ttk.Treeview(tree_frame, columns=('PPM',), show='tree headings', height=15)
        self.tree.heading('#0', text='Элемент')
        self.tree.heading('PPM', text='PPM')
        self.tree.column('#0', width=80)
        self.tree.column('PPM', width=80)
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Результат анализа
        result_frame = tk.LabelFrame(right_frame, text="Результат анализа", padx=5, pady=5)
        result_frame.pack(fill=tk.X, pady=5)

        self.result_text = tk.Text(result_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        self.current_file = None

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите файл с данными РФА",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )
        if not file_path:
            return

        self.current_file = file_path
        self.file_label.config(text=os.path.basename(file_path))
        self.status_var.set("Обработка файла...")
        self.root.update()

        try:
            # Парсинг элементов
            elements = parse_elements_from_file(file_path)
            # Парсинг CPS
            cps, cps_light = parse_cps_blocks(file_path)

            # Обновление таблицы
            for row in self.tree.get_children():
                self.tree.delete(row)
            for elem, ppm in sorted(elements.items(), key=lambda x: x[1], reverse=True):
                self.tree.insert('', 'end', text=elem, values=(f"{ppm:.1f}",))

            # Обновление графика
            self.ax.clear()
            if cps:
                self.ax.plot(cps, label='CPS', color='blue')
            if cps_light:
                self.ax.plot(cps_light, label='CPS Light', color='red')
            if cps or cps_light:
                self.ax.set_title("Дифракционная картина")
                self.ax.set_xlabel("Номер канала")
                self.ax.set_ylabel("Интенсивность")
                self.ax.legend()
                self.ax.grid(True, alpha=0.3)
            else:
                self.ax.text(0.5, 0.5, "Нет данных CPS", ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()

            # Поиск минерала
            if elements and self.mineral_db:
                best = None
                best_score = float('inf')
                for mineral in self.mineral_db:
                    score = compare_composition(elements, mineral['comp'])
                    if score < best_score:
                        best_score = score
                        best = mineral

                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete(1.0, tk.END)
                if best:
                    self.result_text.insert(tk.END, f"Наиболее вероятный минерал:\n{best['name']}\n")
                    self.result_text.insert(tk.END, f"Формула: {best['formula']}\n")
                    self.result_text.insert(tk.END, f"Оценка соответствия: {best_score:.6f}\n")
                    self.result_text.insert(tk.END, "(чем меньше, тем лучше)\n")
                else:
                    self.result_text.insert(tk.END, "Не найдено подходящих минералов.\n")
                self.result_text.config(state=tk.DISABLED)
            else:
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete(1.0, tk.END)
                if not elements:
                    self.result_text.insert(tk.END, "Не удалось извлечь элементный состав из файла.\n")
                else:
                    self.result_text.insert(tk.END, "База минералов не загружена.\n")
                self.result_text.config(state=tk.DISABLED)

            self.status_var.set("Готово")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать файл:\n{str(e)}")
            self.status_var.set("Ошибка")

# -------------------------------------------------------------------
# Запуск приложения
# -------------------------------------------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = RFAApp(root)
    root.mainloop()