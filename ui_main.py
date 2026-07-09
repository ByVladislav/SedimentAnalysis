# ui_main.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import threading
import gc
import pandas as pd
from typing import Optional, List, Tuple, Dict
from collections import defaultdict
import time
from datetime import datetime
import re
import json
import requests
import matplotlib.colors as mcolors

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from config import WAVENUMBERS, WAVENUMBER_RANGE, COLORS, DEEPSEEK_API_KEY, DEEPSEEK_API_URL
from database import DatabaseLoader
from parsers import FileParser
from analyzers import Analyzer


class MineralAnalyzerApp:
    """Главное приложение анализа минералов"""

    def __init__(self, root):
        self.root = root
        self.root.title("Минеральный Анализатор - РФА + Раман")

        # Настройка размера под 2160x1440
        screen_width = 2160
        screen_height = 1440
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.minsize(1200, 800)

        # Переменные
        self.data_folder = None
        self.xrf_spectrum = None
        self.raman_spectrum = None
        self.mineral_db = {}
        self.mineral_by_base_id = {}
        self.mineral_by_name = {}
        self.mineral_by_formula = {}
        self.last_raman_results = None
        self.last_xrf_result = None
        self.analysis_complete = False
        self.all_minerals_list = []
        self.ai_components = None
        self.ai_verdict = None

        # Инициализация загрузчика баз данных
        self.db_loader = DatabaseLoader()

        # Создание интерфейса
        self._setup_ui()

        # Загрузка базы минералов из elements
        self._load_mineral_database()

        # Запускаем загрузку баз данных RRUFF в фоновом режиме
        self.db_loader.load_all(callback=self._update_db_progress)

    def _load_mineral_database(self):
        """Загрузка базы минералов из CSV файлов в папке elements"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        elements_dir = os.path.join(script_dir, "elements")

        if not os.path.exists(elements_dir):
            self.status_var.set("Папка elements не найдена")
            return

        csv_files = [f for f in os.listdir(elements_dir) if f.endswith('.csv')]

        if not csv_files:
            self.status_var.set("CSV файлы в elements не найдены")
            return

        loaded = 0
        for csv_file in csv_files:
            try:
                filepath = os.path.join(elements_dir, csv_file)
                df = pd.read_csv(filepath)

                required_cols = ['id', 'name', 'formula']
                if not all(col in df.columns for col in required_cols):
                    continue

                for _, row in df.iterrows():
                    mineral_id = str(row['id']).strip()
                    if mineral_id and mineral_id != 'nan':
                        name = str(row.get('name', '')).strip()
                        formula = str(row.get('formula', '')).strip()
                        strunz = str(row.get('strunz', '')).strip()

                        mineral_data = {
                            'name': name,
                            'strunz': strunz,
                            'id': mineral_id,
                            'formula': formula,
                            'elements': str(row.get('elements', '')).strip(),
                            'hyperlink': str(row.get('hyperlink', '')).strip(),
                            'wavelength': str(row.get('wavelength', '')).strip(),
                            'status': str(row.get('status', '')).strip(),
                            'comments': str(row.get('comments', '')).strip(),
                            'shift': str(row.get('shift', '')).strip()
                        }

                        self.mineral_db[mineral_id] = mineral_data

                        base_id = mineral_id.split('-')[0] if '-' in mineral_id else mineral_id
                        if base_id not in self.mineral_by_base_id:
                            self.mineral_by_base_id[base_id] = mineral_data.copy()
                            self.mineral_by_base_id[base_id]['id'] = base_id

                        if name:
                            if name not in self.mineral_by_name:
                                self.mineral_by_name[name] = []
                            self.mineral_by_name[name].append(mineral_data)

                        if formula:
                            if formula not in self.mineral_by_formula:
                                self.mineral_by_formula[formula] = []
                            self.mineral_by_formula[formula].append(mineral_data)

                        self.all_minerals_list.append(mineral_data)
                        loaded += 1
            except Exception as e:
                print(f"Ошибка загрузки {csv_file}: {e}")

        print(f"✅ Загружено {loaded} минералов из elements")
        self.status_var.set(f"Загружено {len(self.mineral_db)} минералов из базы elements")

    def _get_mineral_info(self, mineral_id: str) -> Optional[Dict]:
        """Поиск информации о минерале по ID с несколькими попытками"""
        if mineral_id in self.mineral_db:
            return self.mineral_db[mineral_id]

        base_id = mineral_id.split('-')[0] if '-' in mineral_id else mineral_id
        if base_id in self.mineral_by_base_id:
            return self.mineral_by_base_id[base_id]

        for db_id, info in self.mineral_db.items():
            if mineral_id in db_id or db_id in mineral_id:
                return info

        clean_id = re.sub(r'[^a-zA-Z]', '', mineral_id).lower()
        for name, infos in self.mineral_by_name.items():
            clean_name = re.sub(r'[^a-zA-Z]', '', name).lower()
            if clean_id in clean_name or clean_name in clean_id:
                return infos[0]

        return None

    def _find_mineral_by_formula_or_name(self, search_str: str) -> Optional[Dict]:
        """Поиск по формуле или имени"""
        search_str = search_str.strip()
        if not search_str:
            return None

        if search_str in self.mineral_by_formula:
            return self.mineral_by_formula[search_str][0]
        if search_str in self.mineral_by_name:
            return self.mineral_by_name[search_str][0]

        for formula, infos in self.mineral_by_formula.items():
            if search_str in formula or formula in search_str:
                return infos[0]
        for name, infos in self.mineral_by_name.items():
            if search_str in name or name in search_str:
                return infos[0]

        return None

    def _setup_ui(self):
        """Создание пользовательского интерфейса"""
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(top_frame, weight=3)

        bottom_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(bottom_frame, weight=1)

        self._create_top_panel(top_frame)
        self._create_bottom_panel(bottom_frame)
        self._create_status_bar()

    def _create_top_panel(self, parent):
        """Создание верхней панели с управлением и графиками"""
        top_paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        top_paned.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(top_paned)
        top_paned.add(control_frame, weight=1)
        self._create_control_panel(control_frame)

        plot_frame = ttk.Frame(top_paned)
        top_paned.add(plot_frame, weight=4)
        self._create_plot_panel(plot_frame)

    def _create_control_panel(self, parent):
        """Панель управления"""
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # === Загрузка папки ===
        folder_frame = ttk.LabelFrame(scrollable_frame, text="Загрузка данных", padding=10)
        folder_frame.pack(fill=tk.X, pady=5)

        ttk.Button(folder_frame, text="📁 Выбрать папку с данными",
                   command=self._load_data_folder).pack(fill=tk.X, pady=5)

        self.folder_label = ttk.Label(folder_frame, text="Папка не выбрана",
                                      foreground="gray", wraplength=200)
        self.folder_label.pack(fill=tk.X, pady=2)

        self.files_info_label = ttk.Label(folder_frame, text="", foreground="gray")
        self.files_info_label.pack(fill=tk.X, pady=2)

        # === Анализ ===
        analysis_frame = ttk.LabelFrame(scrollable_frame, text="Анализ", padding=10)
        analysis_frame.pack(fill=tk.X, pady=5)

        self.analyze_btn = ttk.Button(analysis_frame, text="▶ Запустить анализ",
                                      command=self._run_analysis, state=tk.DISABLED)
        self.analyze_btn.pack(fill=tk.X, pady=5)

        # Кнопки экспорта
        export_frame = ttk.Frame(analysis_frame)
        export_frame.pack(fill=tk.X, pady=2)

        self.export_btn = ttk.Button(export_frame, text="📤 Экспорт результатов (TXT)",
                                     command=self._export_results, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.export_all_btn = ttk.Button(export_frame, text="📋 Все ID (TXT)",
                                         command=self._export_all_ids, state=tk.NORMAL)
        self.export_all_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Кнопка AI вердикта
        self.ai_btn = ttk.Button(analysis_frame, text="🤖 Получить вердикт AI",
                                 command=self._get_ai_verdict, state=tk.DISABLED)
        self.ai_btn.pack(fill=tk.X, pady=5)

        # === Статус баз данных ===
        db_frame = ttk.LabelFrame(scrollable_frame, text="Базы данных RRUFF", padding=10)
        db_frame.pack(fill=tk.X, pady=5)

        self.db_status_label = ttk.Label(db_frame, text="Загрузка...", foreground="blue")
        self.db_status_label.pack(fill=tk.X, pady=2)

        self.db_stats_label = ttk.Label(db_frame, text="", foreground="gray")
        self.db_stats_label.pack(fill=tk.X, pady=2)

        self.db_progress = ttk.Progressbar(db_frame, mode='determinate', length=200)
        self.db_progress.pack(fill=tk.X, pady=5)

        db_btn_frame = ttk.Frame(db_frame)
        db_btn_frame.pack(fill=tk.X, pady=5)

        self.reload_btn = ttk.Button(db_btn_frame, text="🔄 Перезагрузить базы",
                                     command=self._reload_databases, state=tk.DISABLED)
        self.reload_btn.pack(side=tk.LEFT, padx=2)

        self.cache_info_label = ttk.Label(db_btn_frame, text="", foreground="green", font=('', 8))
        self.cache_info_label.pack(side=tk.LEFT, padx=10)

        # === Элементный состав ===
        elem_frame = ttk.LabelFrame(scrollable_frame, text="Элементный состав (РФА)", padding=10)
        elem_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        tree_container = ttk.Frame(elem_frame)
        tree_container.pack(fill=tk.BOTH, expand=True)

        self.elem_tree = ttk.Treeview(tree_container, columns=('PPM', 'Mass%'),
                                      show='tree headings', height=8)
        self.elem_tree.heading('#0', text='Элемент')
        self.elem_tree.heading('PPM', text='PPM')
        self.elem_tree.heading('Mass%', text='Масс.%')
        self.elem_tree.column('#0', width=70)
        self.elem_tree.column('PPM', width=80)
        self.elem_tree.column('Mass%', width=70)

        tree_scroll = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.elem_tree.yview)
        self.elem_tree.configure(yscrollcommand=tree_scroll.set)

        self.elem_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_plot_panel(self, parent):
        """Панель с графиками"""
        self.fig = Figure(figsize=(10, 7), dpi=100, facecolor='white')
        gs = GridSpec(2, 3, figure=self.fig, hspace=0.35, wspace=0.25)

        self.ax_xrf = self.fig.add_subplot(gs[0, 0])
        self.ax_xrf.set_title("РФА Спектр", fontweight='bold', fontsize=10)
        self.ax_xrf.set_xlabel("Номер канала", fontsize=8)
        self.ax_xrf.set_ylabel("Интенсивность", fontsize=8)
        self.ax_xrf.grid(True, alpha=0.3)
        self.ax_xrf.tick_params(labelsize=8)

        self.ax_raman = self.fig.add_subplot(gs[0, 1])
        self.ax_raman.set_title("Раман Спектр", fontweight='bold', fontsize=10)
        self.ax_raman.set_xlabel("Волновое число (см⁻¹)", fontsize=8)
        self.ax_raman.set_ylabel("Интенсивность (норм.)", fontsize=8)
        self.ax_raman.grid(True, alpha=0.3)
        self.ax_raman.set_xlim(WAVENUMBER_RANGE)
        self.ax_raman.set_ylim(0, 1.1)
        self.ax_raman.tick_params(labelsize=8)

        self.ax_compare = self.fig.add_subplot(gs[0, 2])
        self.ax_compare.set_title("Сравнение с базой", fontweight='bold', fontsize=10)
        self.ax_compare.set_xlabel("Волновое число (см⁻¹)", fontsize=8)
        self.ax_compare.set_ylabel("Интенсивность (норм.)", fontsize=8)
        self.ax_compare.grid(True, alpha=0.3)
        self.ax_compare.set_xlim(WAVENUMBER_RANGE)
        self.ax_compare.set_ylim(0, 1.1)
        self.ax_compare.tick_params(labelsize=8)

        # График для круговой диаграммы (заменяет рейтинг)
        self.ax_combined = self.fig.add_subplot(gs[1, :])
        self.ax_combined.text(0.5, 0.5, "Нажмите 'Получить вердикт AI'\nдля построения диаграммы",
                              ha='center', va='center', fontsize=14, color='gray')
        self.ax_combined.set_title("Распределение компонентов", fontweight='bold', fontsize=12)
        self.ax_combined.axis('off')  # отключаем оси, т.к. это диаграмма

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    def _create_bottom_panel(self, parent):
        """Создание нижней панели с результатами"""
        bottom_paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        bottom_paned.pack(fill=tk.BOTH, expand=True)

        raman_frame = ttk.LabelFrame(bottom_paned, text="Раман совпадения", padding=5)
        bottom_paned.add(raman_frame, weight=1)

        self.raman_results_text = tk.Text(raman_frame, wrap=tk.WORD, font=('Consolas', 10))
        raman_scroll = ttk.Scrollbar(raman_frame, orient=tk.VERTICAL, command=self.raman_results_text.yview)
        self.raman_results_text.configure(yscrollcommand=raman_scroll.set)
        self.raman_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        raman_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        conclusion_frame = ttk.LabelFrame(bottom_paned, text="Технологическое заключение", padding=5)
        bottom_paned.add(conclusion_frame, weight=1)

        self.conclusion_text = tk.Text(conclusion_frame, wrap=tk.WORD, font=('Consolas', 10))
        conclusion_scroll = ttk.Scrollbar(conclusion_frame, orient=tk.VERTICAL, command=self.conclusion_text.yview)
        self.conclusion_text.configure(yscrollcommand=conclusion_scroll.set)
        self.conclusion_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        conclusion_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_status_bar(self):
        """Статусная строка"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar()
        self.status_var.set("Готов к работе")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                 relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_label.pack(fill=tk.X)

    def _export_all_ids(self):
        """Экспорт всех ID из базы elements в TXT файл"""
        if not self.all_minerals_list:
            messagebox.showwarning("Предупреждение", "База элементов не загружена")
            return

        filepath = filedialog.asksaveasfilename(
            title="Сохранить список всех минералов",
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("ВСЕ МИНЕРАЛЫ ИЗ БАЗЫ ELEMENTS\n")
                f.write("=" * 100 + "\n")
                f.write(f"Всего записей: {len(self.all_minerals_list)}\n")
                f.write("=" * 100 + "\n\n")

                f.write(f"{'ID':<15} {'Название':<35} {'Формула':<30} {'Strunz':<15}\n")
                f.write("-" * 100 + "\n")

                sorted_minerals = sorted(self.all_minerals_list, key=lambda x: x['id'])

                for mineral in sorted_minerals:
                    name = mineral.get('name', '')[:35]
                    formula = mineral.get('formula', '')[:30]
                    strunz = mineral.get('strunz', '')[:15]
                    mineral_id = mineral.get('id', '')

                    f.write(f"{mineral_id:<15} {name:<35} {formula:<30} {strunz:<15}\n")

                f.write("\n" + "=" * 100 + "\n")
                f.write("КОНЕЦ СПИСКА\n")
                f.write("=" * 100 + "\n")

            messagebox.showinfo("Экспорт", f"Список всех минералов сохранен в:\n{filepath}")
            self.status_var.set(f"Экспорт всех ID завершен: {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Ошибка экспорта", f"Не удалось сохранить файл:\n{str(e)}")

    def _load_data_folder(self):
        """Загрузка папки с данными"""
        directory = filedialog.askdirectory(title="Выберите папку с данными (x.txt, r.txt)")
        if not directory:
            return

        self.data_folder = directory
        self.folder_label.config(text=os.path.basename(directory), foreground="green")

        xrf_path = os.path.join(directory, "x.txt")
        raman_path = os.path.join(directory, "r.txt")

        files_found = []

        if os.path.exists(xrf_path):
            self.xrf_spectrum = FileParser.parse_xrf_file(xrf_path)
            if self.xrf_spectrum:
                files_found.append("✅ x.txt (РФА)")
                self._update_xrf_plot()
                self._update_elements_table()
            else:
                files_found.append("❌ x.txt (ошибка парсинга)")
        else:
            files_found.append("❌ x.txt (не найден)")

        if os.path.exists(raman_path):
            self.raman_spectrum = FileParser.parse_raman_file(raman_path)
            if self.raman_spectrum is not None:
                files_found.append("✅ r.txt (Раман)")
                self._update_raman_plot()
            else:
                files_found.append("❌ r.txt (ошибка парсинга)")
        else:
            files_found.append("❌ r.txt (не найден)")

        self.files_info_label.config(text=" | ".join(files_found), foreground="blue")
        self._update_analyze_button()

    def _update_db_progress(self, loaded: int, total: int):
        """Обновление прогресса загрузки баз данных"""
        if total > 0:
            percent = (loaded / total) * 100
            self.db_status_label.config(
                text=f"Загрузка: {loaded}/{total} ({percent:.1f}%)",
                foreground="blue"
            )
            self.db_progress['value'] = percent

            stats = self.db_loader.get_stats()
            cache_text = " (из кэша)" if stats.get('from_cache', False) else ""
            loading_text = " (загрузка...)" if stats.get('is_loading', False) else ""
            self.db_stats_label.config(
                text=f"Раман: {len(self.db_loader.raman_db)} спектров | РФА: {len(self.db_loader.xray_db)} минералов{cache_text}{loading_text}",
                foreground="gray"
            )

            if stats.get('from_cache', False):
                self.cache_info_label.config(text="✅ Кэш загружен", foreground="green")
                self.reload_btn.config(state=tk.NORMAL)

            self.root.update_idletasks()
        else:
            self.db_status_label.config(text="Загрузка завершена", foreground="green")
            self.db_progress['value'] = 100

            stats = self.db_loader.get_stats()
            cache_text = " (из кэша)" if stats.get('from_cache', False) else ""
            loading_text = " (загрузка...)" if stats.get('is_loading', False) else ""
            self.db_stats_label.config(
                text=f"Раман: {len(self.db_loader.raman_db)} спектров | РФА: {len(self.db_loader.xray_db)} минералов{cache_text}{loading_text}",
                foreground="gray"
            )

            if stats.get('from_cache', False):
                self.cache_info_label.config(text="✅ Кэш загружен", foreground="green")
                self.reload_btn.config(state=tk.NORMAL)
            else:
                if not stats.get('is_loading', False):
                    self.cache_info_label.config(text="", foreground="green")
                    self.reload_btn.config(state=tk.NORMAL)
                else:
                    self.reload_btn.config(state=tk.DISABLED)

            self._update_analyze_button()

    def _reload_databases(self):
        """Принудительная перезагрузка баз данных"""
        if messagebox.askyesno("Перезагрузка баз",
                               "Перезагрузить базы данных из исходных файлов?\nЭто может занять некоторое время."):
            self.db_status_label.config(text="Перезагрузка...", foreground="orange")
            self.db_progress['value'] = 0
            self.cache_info_label.config(text="⏳ Перезагрузка...", foreground="orange")
            self.reload_btn.config(state=tk.DISABLED)
            self.analyze_btn.config(state=tk.DISABLED)

            self.db_loader.force_reload(callback=self._update_db_progress)

    def _update_analyze_button(self):
        """Обновление состояния кнопки анализа"""
        stats = self.db_loader.get_stats()

        if (self.xrf_spectrum is not None and
                self.raman_spectrum is not None and
                len(self.db_loader.raman_db) > 0):
            self.analyze_btn.config(state=tk.NORMAL)
            if stats.get('is_loading', False):
                self.status_var.set("Базы загружаются (можно анализировать)")
            else:
                self.status_var.set("Готов к анализу")
        elif len(self.db_loader.raman_db) == 0 and not stats.get('is_loading', False):
            self.analyze_btn.config(state=tk.DISABLED)
            self.status_var.set("Базы данных не загружены")
        else:
            self.analyze_btn.config(state=tk.DISABLED)
            self.status_var.set("Загрузите папку с данными")

        # Активация кнопки AI после завершения анализа
        if self.analysis_complete:
            self.ai_btn.config(state=tk.NORMAL)
        else:
            self.ai_btn.config(state=tk.DISABLED)

    def _update_xrf_plot(self):
        """Обновление графика РФА"""
        self.ax_xrf.clear()
        if self.xrf_spectrum:
            if self.xrf_spectrum.cps_data:
                self.ax_xrf.plot(self.xrf_spectrum.cps_data, label='CPS', color='blue', alpha=0.7, linewidth=0.8)
            if self.xrf_spectrum.cps_light_data:
                self.ax_xrf.plot(self.xrf_spectrum.cps_light_data, label='CPS Light', color='red', alpha=0.7,
                                 linewidth=0.8)
            self.ax_xrf.set_title("РФА Спектр", fontweight='bold', fontsize=10)
            self.ax_xrf.set_xlabel("Номер канала", fontsize=8)
            self.ax_xrf.set_ylabel("Интенсивность", fontsize=8)
            if self.xrf_spectrum.cps_data or self.xrf_spectrum.cps_light_data:
                self.ax_xrf.legend(fontsize=8)
            self.ax_xrf.grid(True, alpha=0.3)
            self.ax_xrf.tick_params(labelsize=8)
        else:
            self.ax_xrf.text(0.5, 0.5, "Нет данных", ha='center', va='center',
                             transform=self.ax_xrf.transAxes, fontsize=12)
        self.canvas.draw()

    def _update_raman_plot(self):
        """Обновление графика Рамана"""
        self.ax_raman.clear()
        if self.raman_spectrum is not None:
            self.ax_raman.plot(WAVENUMBERS, self.raman_spectrum, 'b-', linewidth=1.2)
            self.ax_raman.set_title("Раман Спектр", fontweight='bold', fontsize=10)
            self.ax_raman.set_xlabel("Волновое число (см⁻¹)", fontsize=8)
            self.ax_raman.set_ylabel("Интенсивность (норм.)", fontsize=8)
            self.ax_raman.grid(True, alpha=0.3)
            self.ax_raman.set_xlim(WAVENUMBER_RANGE)
            self.ax_raman.set_ylim(0, 1.1)
            self.ax_raman.tick_params(labelsize=8)
        else:
            self.ax_raman.text(0.5, 0.5, "Нет данных", ha='center', va='center',
                               transform=self.ax_raman.transAxes, fontsize=12)
        self.canvas.draw()

    def _update_elements_table(self):
        """Обновление таблицы элементов"""
        for row in self.elem_tree.get_children():
            self.elem_tree.delete(row)

        if self.xrf_spectrum and self.xrf_spectrum.elements:
            total_ppm = sum(self.xrf_spectrum.elements.values())
            for elem, ppm in sorted(self.xrf_spectrum.elements.items(), key=lambda x: x[1], reverse=True)[:20]:
                mass_percent = (ppm / total_ppm * 100) if total_ppm > 0 else 0
                self.elem_tree.insert('', 'end', text=elem, values=(f"{ppm:.1f}", f"{mass_percent:.2f}"))

    def _run_analysis(self):
        """Запуск анализа"""
        if self.raman_spectrum is None:
            messagebox.showwarning("Предупреждение", "Раман-спектр не загружен")
            return

        if self.xrf_spectrum is None:
            messagebox.showwarning("Предупреждение", "РФА-данные не загружены")
            return

        if len(self.db_loader.raman_db) == 0:
            messagebox.showwarning("Предупреждение", "Базы данных не загружены")
            return

        self.status_var.set("Выполняется анализ...")
        self.analyze_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.ai_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._perform_analysis, daemon=True)
        thread.start()

    def _perform_analysis(self):
        """Выполнение анализа в отдельном потоке"""
        try:
            raman_db = self.db_loader.raman_db

            if len(raman_db) == 0:
                self.root.after(0, lambda: self._show_error("База Раман-спектров пуста"))
                return

            self.root.after(0, lambda: self.status_var.set("Сравнение Раман-спектров..."))
            raman_results = Analyzer.compare_raman_spectra_batch(
                self.raman_spectrum,
                raman_db,
                batch_size=2000
            )

            self.root.after(0, lambda: self.status_var.set("Сравнение РФА данных..."))
            xrf_result = Analyzer.find_best_xray_match(
                self.xrf_spectrum.elements,
                self.db_loader.xray_db
            )

            self.last_raman_results = raman_results
            self.last_xrf_result = xrf_result

            gc.collect()

            self.root.after(0, lambda: self._display_results(raman_results, xrf_result))

        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
        finally:
            gc.collect()

    def _group_results_by_id(self, raman_results):
        """Группировка результатов по ID с подсчетом количества совпадений"""
        grouped = defaultdict(lambda: {'count': 0, 'max_similarity': 0, 'sample_ids': [], 'mineral_info': None})

        for sample_id, sim in raman_results:
            if sim > 0.15:
                rruff_id = sample_id.split('__')[1] if '__' in sample_id else sample_id
                mineral_info = self._get_mineral_info(rruff_id)

                if not mineral_info:
                    name_part = sample_id.split('__')[0] if '__' in sample_id else sample_id
                    mineral_info = self._find_mineral_by_formula_or_name(name_part.replace('_', ' '))

                grouped[rruff_id]['count'] += 1
                grouped[rruff_id]['max_similarity'] = max(grouped[rruff_id]['max_similarity'], sim)
                grouped[rruff_id]['sample_ids'].append(sample_id)
                if mineral_info:
                    grouped[rruff_id]['mineral_info'] = mineral_info

        result = []
        for rruff_id, data in grouped.items():
            mineral_info = data.get('mineral_info')
            result.append({
                'id': rruff_id,
                'count': data['count'],
                'max_similarity': data['max_similarity'],
                'sample_ids': data['sample_ids'],
                'formula': mineral_info.get('formula', '') if mineral_info else '',
                'name': mineral_info.get('name', '') if mineral_info else '',
                'mineral_info': mineral_info
            })

        result.sort(key=lambda x: x['max_similarity'], reverse=True)
        return result

    def _display_results(self, raman_results, xrf_result):
        """Отображение результатов с группировкой по ID"""
        self.raman_results_text.delete(1.0, tk.END)
        self.conclusion_text.delete(1.0, tk.END)

        grouped_results = self._group_results_by_id(raman_results)

        self.raman_results_text.insert(tk.END, "=" * 80 + "\n")
        self.raman_results_text.insert(tk.END, "ТОП СОВПАДЕНИЙ ПО РАМАНУ (сгруппировано по ID)\n")
        self.raman_results_text.insert(tk.END, "=" * 80 + "\n\n")

        comparison_data = []
        for idx, item in enumerate(grouped_results[:10], 1):
            rruff_id = item['id']
            count = item['count']
            sim = item['max_similarity']
            formula = item['formula']
            name = item['name']
            mineral_info = item['mineral_info']

            if formula:
                display_name = formula
            elif name:
                display_name = name
            else:
                display_name = rruff_id

            underline = "~" * len(display_name) if formula else ""

            bar = "█" * int(sim * 30) + "░" * (30 - int(sim * 30))
            line = f"{idx:2}. {display_name:<30} {sim * 100:5.1f}% {bar}\n"
            if underline:
                line += f"    {underline}\n"
            line += f"    ID: {rruff_id} | Совпадений: {count}\n"

            if mineral_info:
                if mineral_info.get('elements'):
                    line += f"    Элементы: {mineral_info.get('elements')}\n"
                if mineral_info.get('strunz'):
                    line += f"    Strunz: {mineral_info.get('strunz')}\n"
                if mineral_info.get('comments'):
                    line += f"    Комментарии: {mineral_info.get('comments')}\n"
            else:
                line += f"    (Информация не найдена в базе elements)\n"

            self.raman_results_text.insert(tk.END, line)
            comparison_data.append((rruff_id, sim, count, formula, name))

        if xrf_result:
            mineral = xrf_result[0]
            score = xrf_result[1]
            self.raman_results_text.insert(tk.END, "\n" + "=" * 80 + "\n")
            self.raman_results_text.insert(tk.END, "РЕЗУЛЬТАТ РФА АНАЛИЗА\n")
            self.raman_results_text.insert(tk.END, "=" * 80 + "\n")
            self.raman_results_text.insert(tk.END, f"Минерал: {mineral['name']}\n")
            self.raman_results_text.insert(tk.END, f"Формула: {mineral['formula']}\n")
            self.raman_results_text.insert(tk.END, f"Оценка соответствия: {score:.6f}\n")

        self._update_comparison_plot(comparison_data)
        # Вместо рейтинга показываем заглушку для диаграммы
        self._show_waiting_for_ai()
        self._generate_conclusion(grouped_results, xrf_result)

        self.export_btn.config(state=tk.NORMAL)
        self.analysis_complete = True
        self.ai_btn.config(state=tk.NORMAL)

        gc.collect()
        self.status_var.set("Анализ завершен")
        self.analyze_btn.config(state=tk.NORMAL)

    def _show_waiting_for_ai(self):
        """Показывает сообщение ожидания на месте диаграммы"""
        self.ax_combined.clear()
        self.ax_combined.text(0.5, 0.5, "Нажмите 'Получить вердикт AI'\nдля построения диаграммы",
                              ha='center', va='center', fontsize=14, color='gray')
        self.ax_combined.set_title("Распределение компонентов", fontweight='bold', fontsize=12)
        self.ax_combined.axis('off')
        self.canvas.draw()

    def _update_comparison_plot(self, comparison_data):
        """Обновление графика сравнения"""
        self.ax_compare.clear()
        self.ax_compare.plot(WAVENUMBERS, self.raman_spectrum, 'b-', linewidth=2, label='Образец', alpha=0.8)

        for i, (rruff_id, sim, count, formula, name) in enumerate(comparison_data[:4]):
            if sim > 0.15:
                sample_id = None
                for sid, s in self.last_raman_results:
                    current_id = sid.split('__')[1] if '__' in sid else sid
                    if current_id == rruff_id:
                        sample_id = sid
                        break

                if sample_id and sample_id in self.db_loader.raman_db:
                    ref = self.db_loader.raman_db[sample_id]
                    color = COLORS[i % len(COLORS)]
                    label = formula if formula else name if name else rruff_id
                    self.ax_compare.plot(WAVENUMBERS, ref, color, linewidth=1.2,
                                         linestyle='--', alpha=0.6,
                                         label=f"{label[:20]} ({sim * 100:.1f}%, {count} совп.)")

        self.ax_compare.set_title("Сравнение с базой", fontweight='bold', fontsize=10)
        self.ax_compare.set_xlabel("Волновое число (см⁻¹)", fontsize=8)
        self.ax_compare.set_ylabel("Интенсивность (норм.)", fontsize=8)
        self.ax_compare.grid(True, alpha=0.3)
        self.ax_compare.set_xlim(WAVENUMBER_RANGE)
        self.ax_compare.set_ylim(0, 1.1)
        if comparison_data:
            self.ax_compare.legend(loc='upper right', fontsize=7)
        self.ax_compare.tick_params(labelsize=8)
        self.canvas.draw()

    def _generate_conclusion(self, grouped_results, xrf_result):
        """Генерация автоматического заключения"""
        conclusion = "=" * 60 + "\n"
        conclusion += "ТЕХНОЛОГИЧЕСКОЕ ЗАКЛЮЧЕНИЕ (АВТОМАТИЧЕСКОЕ)\n"
        conclusion += "=" * 60 + "\n\n"

        top_mineral = None
        top_similarity = 0
        top_id = None
        top_count = 0
        top_formula = ""
        top_name = ""

        if grouped_results:
            top_mineral = grouped_results[0]
            top_id = top_mineral['id']
            top_similarity = top_mineral['max_similarity']
            top_count = top_mineral['count']
            top_formula = top_mineral['formula']
            top_name = top_mineral['name']

        mineral_info = top_mineral.get('mineral_info') if top_mineral else None

        xrf_match = False
        if xrf_result and mineral_info:
            xrf_mineral = xrf_result[0]['name'].lower()
            mineral_name = mineral_info.get('name', '').lower()
            if mineral_name and (xrf_mineral in mineral_name or mineral_name in xrf_mineral):
                xrf_match = True

        if mineral_info and top_similarity > 0.5:
            display_name = top_formula if top_formula else mineral_info.get('name', 'Unknown')
            conclusion += f"🔬 ОПРЕДЕЛЕН МИНЕРАЛ: {display_name}\n"
            if top_formula:
                conclusion += f"   Название: {mineral_info.get('name', '')}\n"
            conclusion += f"   ID: {top_id}\n"
            conclusion += f"   Формула: {mineral_info.get('formula', '')}\n"
            conclusion += f"   Strunz: {mineral_info.get('strunz', '')}\n"
            conclusion += f"   Достоверность: {top_similarity * 100:.1f}%\n"
            conclusion += f"   Совпадений в базе: {top_count}\n"
            if mineral_info.get('elements'):
                conclusion += f"   Элементы: {mineral_info.get('elements')}\n"
            if mineral_info.get('hyperlink'):
                conclusion += f"   Ссылка: {mineral_info.get('hyperlink')}\n"
            if mineral_info.get('comments'):
                conclusion += f"   Комментарии: {mineral_info.get('comments')}\n"
            if xrf_match:
                conclusion += "   ✅ РФА подтверждает определение\n"
            else:
                conclusion += "   ⚠️ РФА не подтверждает (возможна примесь)\n"
        elif mineral_info and top_similarity > 0.3:
            display_name = top_formula if top_formula else mineral_info.get('name', 'Unknown')
            conclusion += f"🔬 ПРЕДПОЛОЖИТЕЛЬНЫЙ МИНЕРАЛ: {display_name}\n"
            conclusion += f"   ID: {top_id}\n"
            conclusion += f"   Формула: {mineral_info.get('formula', '')}\n"
            conclusion += f"   Достоверность: {top_similarity * 100:.1f}%\n"
            conclusion += f"   Совпадений в базе: {top_count}\n"
            conclusion += "   ⚠️ Требуется дополнительное подтверждение\n"
        else:
            conclusion += "❌ ОПРЕДЕЛИТЬ МИНЕРАЛ НЕ УДАЛОСЬ\n"
            if top_id:
                conclusion += f"   Ближайший ID: {top_id}\n"
                if top_formula:
                    conclusion += f"   Формула: {top_formula}\n"
            conclusion += "   Специфическая аморфная или органическая матрица\n"

        conclusion += "\n" + "-" * 60 + "\n\n"
        conclusion += "ОЦЕНКА ТЕХНОЛОГИЧЕСКИХ РИСКОВ:\n\n"

        risk_found = False
        if mineral_info:
            mineral_name = mineral_info.get('name', '').lower()
            if any(x in mineral_name for x in ['halite', 'sylvite', 'kcl', 'nacl']):
                conclusion += "  ⚠️ Соли/галогениды → Риск солеотложения и коррозии\n"
                conclusion += "     Рекомендация: ингибитор солеотложения\n"
                risk_found = True
            elif any(x in mineral_name for x in ['gypsum', 'anhydrite', 'barite', 'celestine']):
                conclusion += "  ⚠️ Сульфаты → Риск образования осадка в трубах\n"
                conclusion += "     Рекомендация: фосфонаты, антискаланты\n"
                risk_found = True
            elif any(x in mineral_name for x in ['kaolinite', 'illite', 'montmorillonite', 'smectite']):
                conclusion += "  ⚠️ Глинистые минералы → Риск набухания пласта\n"
                conclusion += "     Рекомендация: ингибированные растворы (KCl)\n"
                risk_found = True
            elif any(x in mineral_name for x in ['quartz', 'feldspar', 'calcite', 'dolomite']):
                conclusion += "  ✅ Стабильная матрица\n"
                conclusion += "     Агрессивных осложнений не прогнозируется\n"
                risk_found = True

        if not risk_found:
            conclusion += "  ✅ Матрица стабильна\n"
            conclusion += "     Химически агрессивных осложнений не прогнозируется\n"

        conclusion += "\n" + "=" * 60 + "\n"
        conclusion += f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"

        self.conclusion_text.insert(1.0, conclusion)

    def _get_ai_verdict(self):
        """Получение вердикта от DeepSeek API"""
        if not self.analysis_complete:
            messagebox.showwarning("Предупреждение", "Сначала выполните анализ")
            return

        api_key = DEEPSEEK_API_KEY.strip()
        if not api_key:
            key = simpledialog.askstring("API Key", "Введите ваш API ключ DeepSeek:", show='*')
            if not key:
                return
            api_key = key.strip()

        self.status_var.set("Запрос к DeepSeek API...")
        self.ai_btn.config(state=tk.DISABLED)

        report = self._prepare_report_for_ai()
        if not report:
            self.status_var.set("Ошибка подготовки отчёта")
            self.ai_btn.config(state=tk.NORMAL)
            return

        prompt = self._build_ai_prompt(report)

        thread = threading.Thread(target=self._call_deepseek_api, args=(api_key, prompt), daemon=True)
        thread.start()

    def _prepare_report_for_ai(self):
        """Формирует текстовый отчёт для отправки в AI"""
        lines = []
        lines.append("РФА СПЕКТР:")
        if self.xrf_spectrum and self.xrf_spectrum.elements:
            total_ppm = sum(self.xrf_spectrum.elements.values())
            for elem, ppm in sorted(self.xrf_spectrum.elements.items(), key=lambda x: x[1], reverse=True):
                mass = (ppm / total_ppm * 100) if total_ppm > 0 else 0
                lines.append(f"  {elem}: {ppm:.1f} ppm ({mass:.2f} масс.%)")

        lines.append("\nРАМАН АНАЛИЗ:")
        if self.last_raman_results:
            grouped = self._group_results_by_id(self.last_raman_results)
            for i, item in enumerate(grouped[:10]):
                name = item.get('name', item['id'])
                formula = item.get('formula', '')
                sim = item['max_similarity'] * 100
                lines.append(f"  {i + 1}. {name} ({formula}) – {sim:.1f}%")

        lines.append("\nАВТОМАТИЧЕСКОЕ ЗАКЛЮЧЕНИЕ:")
        lines.append(self.conclusion_text.get(1.0, tk.END).strip())

        return "\n".join(lines)

    def _build_ai_prompt(self, report_text):
        """Формирует промт для DeepSeek"""
        return f"""
Ты – эксперт по минералогическому анализу, специализирующийся на совместной интерпретации данных рентгено-флуоресцентного анализа (РФА) и Раман-спектроскопии.

На вход подаётся текстовый отчёт, содержащий:
1. Блок «РФА СПЕКТР» – элементный состав в ppm и масс.%, где перечислены элементы и их концентрации.
2. Блок «РАМАН АНАЛИЗ» – список совпадений с базой RRUFF, включающий ID минерала, название, формулу, процент сходства, элементы и комментарии.
3. Блок «ЗАКЛЮЧЕНИЕ» – автоматическое заключение программы, но оно может быть ошибочным (например, программа выбирает минерал, не соответствующий РФА). Твоя задача – перепроверить и выдать собственный обоснованный вердикт.

Правила анализа:
- РФА даёт количественные массовые доли для тяжёлых элементов (Z > 10), но НЕ видит лёгкие элементы (O, H, F, а также Na может маскироваться под Mg). Это означает, что масс.% в РФА нормированы только на детектируемые элементы и завышены относительно реального содержания в минерале. Однако они надёжно показывают, какие элементы доминируют.
- Раман даёт качественный состав и возможные минеральные фазы, но его количественная оценка ненадёжна (процент сходства – это совпадение спектра, а не содержание). Тем не менее, он помогает идентифицировать конкретные соединения.
- Основным минералом следует считать тот, который:
  - содержит доминирующие элементы РФА (один или несколько);
  - имеет высокое совпадение в Рамане (желательно > 80–85%) и находится в топе списка;
  - при противоречии (например, топ Рамана – Creedite, а РФА показывает много Ba и S) приоритет отдаётся РФА: основным будет минерал, объясняющий основные элементы (в данном случае – барит).
- Если в РФА присутствует значительное количество элемента, не входящего в основной минерал (например, Mg в образце с гипсом), его следует относить к отдельной примесной фазе (например, магниевые сульфаты).
- Все остальные элементы с концентрацией менее ~1% (или не входящие в основной минерал) объединяются в группу «прочие примеси» и перечисляются в таблице.
- Для каждого компонента (основной минерал, значительные примеси, прочие примеси) нужно указать оценочную массовую долю в процентах от общей массы пробы. Доли должны быть разумными, исходя из РФА и стехиометрии (например, если Ba и S в барите, то почти весь Ba идёт в BaSO4, и его массовая доля ≈ масс.% Ba / (молярная доля Ba в BaSO4) – можно дать приблизительную оценку). Допускается округление до целых или десятых.
- В ответе не нужно разделять основной минерал на отдельные ионы (например, K⁺ и NO₃⁻ выводить как единое соединение KNO₃, а не разделять).

Ответ предоставь строго в формате JSON со следующей структурой:
{{
  "conclusion": "краткое текстовое заключение (2–3 предложения)",
  "components": [
    {{
      "name": "название минерала или группы",
      "formula": "химическая формула",
      "percent": число (доля в процентах),
      "source": "РФА / Раман / оба",
      "color": "название цвета на английском (например, blue, red, green) или HEX-код"
    }}
  ],
  "functional_groups": "строка с перечислением основных функциональных групп (например, сульфаты, карбонаты, оксиды) и примесей, которые были выявлены или предположены на основе РФА и Рамана. Укажите также, какие группы являются доминирующими, а какие – второстепенными.",
  "notes": "дополнительные примечания (если есть)"
}}
Сумма процентов должна быть близка к 100% (в пределах нескольких процентов). Цвета выбери понятные и контрастные.

Теперь обработай следующий отчёт и выдай результат в JSON:

{report_text}
"""

    def _call_deepseek_api(self, api_key, prompt):
        """Вызов DeepSeek API"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                ai_message = result['choices'][0]['message']['content']
                try:
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', ai_message)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = ai_message
                    verdict = json.loads(json_str)
                    self.root.after(0, lambda: self._apply_ai_verdict(verdict))
                except json.JSONDecodeError:
                    self.root.after(0, lambda: self._show_error(f"Не удалось распарсить ответ AI:\n{ai_message[:500]}"))
            else:
                error_msg = response.text if response.text else "Неизвестная ошибка"
                self.root.after(0, lambda: self._show_error(f"Ошибка API: {response.status_code}\n{error_msg}"))
        except Exception as e:
            self.root.after(0, lambda: self._show_error(f"Ошибка запроса к API: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.ai_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("Готово"))

    def _apply_ai_verdict(self, verdict):
        """Применение вердикта AI: обновляем заключение и строим круговую диаграмму"""
        conclusion_text = verdict.get('conclusion', '')
        notes = verdict.get('notes', '')
        functional_groups = verdict.get('functional_groups', '')
        components = verdict.get('components', [])

        if conclusion_text or functional_groups or components:
            self.conclusion_text.delete(1.0, tk.END)
            self.conclusion_text.insert(tk.END, "ЭКСПЕРТНЫЙ ВЕРДИКТ (AI):\n")
            self.conclusion_text.insert(tk.END, "=" * 60 + "\n\n")

            if conclusion_text:
                self.conclusion_text.insert(tk.END, conclusion_text + "\n\n")

            if functional_groups:
                self.conclusion_text.insert(tk.END, "ФУНКЦИОНАЛЬНЫЕ ГРУППЫ И ПРИМЕСИ:\n")
                self.conclusion_text.insert(tk.END, functional_groups + "\n\n")

            if components:
                self.conclusion_text.insert(tk.END, "РАСПРЕДЕЛЕНИЕ КОМПОНЕНТОВ:\n")
                self.conclusion_text.insert(tk.END,
                                            f"{'Компонент':<25} {'Формула':<20} {'Доля, %':<10} {'Источник':<10}\n")
                self.conclusion_text.insert(tk.END, "-" * 70 + "\n")
                for comp in components:
                    name = comp.get('name', '')
                    formula = comp.get('formula', '')
                    percent = comp.get('percent', 0)
                    source = comp.get('source', '')
                    self.conclusion_text.insert(tk.END, f"{name:<25} {formula:<20} {percent:<10.1f} {source:<10}\n")
                self.conclusion_text.insert(tk.END, "\n")

            if notes:
                self.conclusion_text.insert(tk.END, "Примечания:\n" + notes + "\n")

            self.conclusion_text.insert(tk.END, "\n" + "=" * 60 + "\n")
            self.conclusion_text.insert(tk.END, f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")

        if components:
            self._draw_pie_chart(components)
            self.ai_components = components

        self.status_var.set("Вердикт AI получен")

    def _draw_pie_chart(self, components):
        """Рисует круговую диаграмму на месте рейтинга совпадений"""
        self.ax_combined.clear()
        if not components:
            self.ax_combined.text(0.5, 0.5, "Нет данных для диаграммы", ha='center', va='center', fontsize=14)
            self.ax_combined.axis('off')
            self.canvas.draw()
            return

        labels = []
        sizes = []
        colors = []
        explode = []

        total = sum(comp.get('percent', 0) for comp in components)
        if total == 0:
            self.ax_combined.text(0.5, 0.5, "Некорректные данные", ha='center', va='center', fontsize=14)
            self.ax_combined.axis('off')
            self.canvas.draw()
            return

        for i, comp in enumerate(components):
            name = comp.get('name', '')
            formula = comp.get('formula', '')
            percent = comp.get('percent', 0)
            color = comp.get('color', COLORS[i % len(COLORS)])

            # Проверяем, является ли color допустимым цветом
            if isinstance(color, str) and mcolors.is_color_like(color):
                colors.append(color)
            else:
                colors.append(COLORS[i % len(COLORS)])

            label = f"{name}\n{formula}" if formula else name
            labels.append(label)
            sizes.append(percent)
            explode.append(0.05 if i == 0 else 0)

        # Включаем оси для pie
        self.ax_combined.axis('on')
        wedges, texts, autotexts = self.ax_combined.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            explode=explode,
            startangle=90,
            shadow=True,
            textprops={'fontsize': 10}
        )
        self.ax_combined.set_title("Распределение компонентов (вердикт AI)", fontweight='bold', fontsize=12)
        self.ax_combined.legend(wedges, labels,
                                title="Компоненты",
                                loc="center left",
                                bbox_to_anchor=(1, 0, 0.5, 1))
        self.canvas.draw()

    def _export_results(self):
        """Экспорт результатов анализа в TXT файл"""
        if not self.analysis_complete:
            messagebox.showwarning("Предупреждение", "Сначала выполните анализ")
            return

        if not self.last_raman_results:
            messagebox.showwarning("Предупреждение", "Нет результатов для экспорта")
            return

        filepath = filedialog.asksaveasfilename(
            title="Сохранить результаты анализа",
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("РЕЗУЛЬТАТЫ АНАЛИЗА МИНЕРАЛА\n")
                f.write("=" * 80 + "\n")
                f.write(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
                f.write("=" * 80 + "\n\n")

                f.write("-" * 80 + "\n")
                f.write("1. РФА СПЕКТР\n")
                f.write("-" * 80 + "\n")

                if self.xrf_spectrum and self.xrf_spectrum.elements:
                    f.write("\nЭлементный состав (PPM):\n")
                    total_ppm = sum(self.xrf_spectrum.elements.values())
                    for elem, ppm in sorted(self.xrf_spectrum.elements.items(), key=lambda x: x[1], reverse=True):
                        mass_percent = (ppm / total_ppm * 100) if total_ppm > 0 else 0
                        f.write(f"  {elem}: {ppm:.1f} PPM ({mass_percent:.2f} масс.%)\n")

                    if self.xrf_spectrum.cps_data:
                        f.write(f"\nCPS данные: {len(self.xrf_spectrum.cps_data)} точек\n")
                    if self.xrf_spectrum.cps_light_data:
                        f.write(f"CPS Light данные: {len(self.xrf_spectrum.cps_light_data)} точек\n")

                if self.last_xrf_result:
                    mineral = self.last_xrf_result[0]
                    score = self.last_xrf_result[1]
                    f.write("\nРезультат РФА анализа:\n")
                    f.write(f"  Минерал: {mineral['name']}\n")
                    f.write(f"  Формула: {mineral['formula']}\n")
                    f.write(f"  Оценка соответствия: {score:.6f}\n")

                f.write("\n" + "-" * 80 + "\n")
                f.write("2. РАМАН АНАЛИЗ\n")
                f.write("-" * 80 + "\n")

                grouped_results = self._group_results_by_id(self.last_raman_results)

                f.write(f"\nВсего совпадений: {len(self.last_raman_results)}\n")
                f.write(f"Уникальных ID: {len(grouped_results)}\n\n")

                f.write("ТОП СОВПАДЕНИЙ ПО ID:\n")
                f.write("-" * 60 + "\n")

                for idx, item in enumerate(grouped_results[:20], 1):
                    rruff_id = item['id']
                    count = item['count']
                    sim = item['max_similarity']
                    formula = item['formula']
                    name = item['name']
                    mineral_info = item['mineral_info']

                    f.write(f"\n{idx}. {formula if formula else name if name else rruff_id}\n")
                    f.write(f"   ID: {rruff_id}\n")
                    f.write(f"   Совпадений: {count}\n")
                    f.write(f"   Макс. сходство: {sim * 100:.1f}%\n")

                    if mineral_info:
                        f.write(f"   Название: {mineral_info.get('name', '')}\n")
                        f.write(f"   Формула: {mineral_info.get('formula', '')}\n")
                        f.write(f"   Strunz: {mineral_info.get('strunz', '')}\n")
                        f.write(f"   Элементы: {mineral_info.get('elements', '')}\n")
                        f.write(f"   Wavelength: {mineral_info.get('wavelength', '')}\n")
                        f.write(f"   Status: {mineral_info.get('status', '')}\n")
                        f.write(f"   Shift: {mineral_info.get('shift', '')}\n")
                        f.write(f"   Comments: {mineral_info.get('comments', '')}\n")
                        if mineral_info.get('hyperlink'):
                            f.write(f"   Ссылка: {mineral_info.get('hyperlink')}\n")
                    else:
                        f.write("   (Информация о минерале не найдена в базе elements)\n")

                f.write("\n" + "-" * 80 + "\n")
                f.write("3. ЗАКЛЮЧЕНИЕ\n")
                f.write("-" * 80 + "\n")
                f.write(self.conclusion_text.get(1.0, tk.END))

                f.write("\n" + "=" * 80 + "\n")
                f.write("КОНЕЦ ОТЧЕТА\n")
                f.write("=" * 80 + "\n")

            messagebox.showinfo("Экспорт", f"Результаты сохранены в:\n{filepath}")
            self.status_var.set(f"Экспорт завершен: {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Ошибка экспорта", f"Не удалось сохранить файл:\n{str(e)}")

    def _show_error(self, error_msg: str):
        """Отображение ошибки"""
        messagebox.showerror("Ошибка", error_msg)
        self.status_var.set("Ошибка")
        self.ai_btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = MineralAnalyzerApp(root)
    root.mainloop()