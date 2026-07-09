import os
import zipfile
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.interpolate import interp1d
from scipy.spatial.distance import cosine
from io import StringIO
import threading

# Глобальные переменные
wavenumbers = np.linspace(100, 1500, 1000)
global_database = {}


class RamanAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Глобальная поисковая система НТЦ - Анализ Раман-спектров")
        self.root.geometry("1400x900")

        # Переменные для файлов
        self.esp_file_path = None
        self.xrf_file_path = None
        self.target_spectrum = None

        # Создание интерфейса
        self.create_widgets()

        # Загрузка баз данных в фоновом режиме
        self.status_label.config(text="Загрузка баз данных...")
        threading.Thread(target=self.load_databases, daemon=True).start()

    def create_widgets(self):
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Настройка весов для ресайза
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Левая панель управления
        left_frame = ttk.LabelFrame(main_frame, text="Управление", padding="10")
        left_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Кнопка загрузки спектра
        ttk.Button(left_frame, text="Загрузить спектр (.esp/.txt)",
                   command=self.load_spectrum).grid(row=0, column=0, pady=5, sticky=tk.W)
        self.spectrum_label = ttk.Label(left_frame, text="Файл не выбран", wraplength=200)
        self.spectrum_label.grid(row=1, column=0, pady=2, sticky=tk.W)

        # Кнопка загрузки РФА
        ttk.Button(left_frame, text="Загрузить РФА (.csv)",
                   command=self.load_xrf).grid(row=2, column=0, pady=5, sticky=tk.W)
        self.xrf_label = ttk.Label(left_frame, text="Файл не выбран", wraplength=200)
        self.xrf_label.grid(row=3, column=0, pady=2, sticky=tk.W)

        # Кнопка анализа
        ttk.Button(left_frame, text="▶ Анализировать",
                   command=self.analyze_sample).grid(row=4, column=0, pady=15, sticky=tk.W)

        # Статус загрузки баз данных
        ttk.Label(left_frame, text="Статус баз данных:").grid(row=5, column=0, pady=(20, 0), sticky=tk.W)
        self.status_label = ttk.Label(left_frame, text="Инициализация...", foreground="blue")
        self.status_label.grid(row=6, column=0, pady=2, sticky=tk.W)

        # Прогресс загрузки
        self.progress_bar = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress_bar.grid(row=7, column=0, pady=10, sticky=(tk.W, tk.E))

        # Результаты поиска
        results_frame = ttk.LabelFrame(left_frame, text="Топ совпадений", padding="5")
        results_frame.grid(row=8, column=0, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        left_frame.rowconfigure(8, weight=1)

        self.results_text = tk.Text(results_frame, height=10, width=30)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Панель для графиков (правая часть)
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        # Создание фигуры matplotlib
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.fig.tight_layout(pad=3.0)

        # Настройка первого графика (спектр образца)
        self.ax1.set_title("Спектр образца", fontweight='bold')
        self.ax1.set_xlabel("Волновое число (см⁻¹)")
        self.ax1.set_ylabel("Интенсивность (норм.)")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(100, 1500)
        self.ax1.set_ylim(0, 1.1)

        # Настройка второго графика (сравнение)
        self.ax2.set_title("Сравнение с базами данных", fontweight='bold')
        self.ax2.set_xlabel("Волновое число (см⁻¹)")
        self.ax2.set_ylabel("Интенсивность (норм.)")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlim(100, 1500)
        self.ax2.set_ylim(0, 1.1)

        # Встраивание графика в tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Панель инструментов matplotlib
        toolbar_frame = ttk.Frame(main_frame)
        toolbar_frame.grid(row=1, column=1, sticky=(tk.W, tk.E))
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        # Панель отчета (нижняя часть)
        report_frame = ttk.LabelFrame(main_frame, text="Лабораторный рапорт НТЦ", padding="10")
        report_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        report_frame.columnconfigure(0, weight=1)
        report_frame.rowconfigure(0, weight=1)

        self.report_text = scrolledtext.ScrolledText(report_frame, height=12, wrap=tk.WORD)
        self.report_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def load_databases(self):
        """Загрузка всех баз данных из папки raw_rruff/raman"""
        try:
            self.progress_bar.start()
            database_path = "raw_rruff/raman"

            if not os.path.exists(database_path):
                self.update_status("Ошибка: папка raw_rruff/raman не найдена", "red")
                return

            zip_files = [f for f in os.listdir(database_path) if f.endswith('.zip')]

            if not zip_files:
                self.update_status("Ошибка: ZIP-файлы баз данных не найдены", "red")
                return

            total_loaded = 0
            for i, zip_file in enumerate(zip_files):
                zip_path = os.path.join(database_path, zip_file)
                self.update_status(f"Загрузка {zip_file} ({i + 1}/{len(zip_files)})...")

                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        for filename in zip_ref.namelist():
                            if filename.endswith('.txt'):
                                sample_id = filename.replace('.txt', '')
                                try:
                                    with zip_ref.open(filename) as f:
                                        content = f.read().decode('utf-8', errors='ignore')

                                    # Парсинг спектра
                                    x_data, y_data = [], []
                                    for line in content.splitlines():
                                        cleaned = line.strip()
                                        if not cleaned or cleaned.startswith('#') or cleaned[0].isalpha():
                                            continue
                                        parts = cleaned.replace(',', ' ').split()
                                        if len(parts) >= 2:
                                            try:
                                                x_data.append(float(parts[0]))
                                                y_data.append(float(parts[1]))
                                            except ValueError:
                                                pass

                                    if len(x_data) > 10 and len(x_data) == len(y_data):
                                        x_arr = np.array(x_data, dtype=float)
                                        y_arr = np.array(y_data, dtype=float)
                                        sort_idx = np.argsort(x_arr)
                                        x_arr = x_arr[sort_idx]
                                        y_arr = y_arr[sort_idx]

                                        _, unique_indices = np.unique(x_arr, return_index=True)
                                        x_arr = x_arr[unique_indices]
                                        y_arr = y_arr[unique_indices]

                                        if len(x_arr) > 10:
                                            f_int = interp1d(x_arr, y_arr, bounds_error=False, fill_value=0.0)
                                            interpolated_y = f_int(wavenumbers)
                                            max_val = np.max(interpolated_y)
                                            if max_val > 0:
                                                global_database[sample_id] = interpolated_y / max_val
                                                total_loaded += 1
                                except Exception as e:
                                    continue
                except Exception as e:
                    print(f"Ошибка чтения {zip_file}: {e}")

            self.update_status(f"✓ Загружено {total_loaded} спектров из {len(zip_files)} баз", "green")

        except Exception as e:
            self.update_status(f"Ошибка: {str(e)}", "red")
        finally:
            self.progress_bar.stop()

    def update_status(self, message, color="blue"):
        """Обновление статусной строки"""
        self.root.after(0, lambda: self.status_label.config(text=message, foreground=color))

    def load_spectrum(self):
        """Загрузка файла спектра"""
        filename = filedialog.askopenfilename(
            title="Выберите файл спектра",
            filetypes=[("Spectrum files", "*.esp *.txt"), ("All files", "*.*")]
        )
        if filename:
            self.esp_file_path = filename
            self.spectrum_label.config(text=os.path.basename(filename))
            # Сразу парсим для предпросмотра
            self.parse_and_plot_spectrum()

    def load_xrf(self):
        """Загрузка файла РФА"""
        filename = filedialog.askopenfilename(
            title="Выберите файл РФА",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.xrf_file_path = filename
            self.xrf_label.config(text=os.path.basename(filename))

    def parse_esp_file(self, filepath):
        """Парсинг файла спектра"""
        x_data, y_data = [], []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            for line in content.splitlines():
                cleaned = line.strip()
                if not cleaned or cleaned[0].isalpha() or cleaned.startswith('<') or \
                        cleaned.startswith('[') or cleaned.startswith(';'):
                    continue
                parts = cleaned.replace(',', ' ').replace('\t', ' ').split()
                if len(parts) >= 2:
                    try:
                        x_data.append(float(parts[0]))
                        y_data.append(float(parts[1]))
                    except ValueError:
                        pass

            if len(x_data) < 10:
                return None

            xa = np.array(x_data, dtype=float)
            ya = np.array(y_data, dtype=float)
            sort_idx = np.argsort(xa)
            xa = xa[sort_idx]
            ya = ya[sort_idx]

            _, unique_indices = np.unique(xa, return_index=True)
            xa = xa[unique_indices]
            ya = ya[unique_indices]

            f_int = interp1d(xa, ya, bounds_error=False, fill_value=0.0)
            interp_y = f_int(wavenumbers)
            max_val = np.max(interp_y)

            if max_val > 0:
                return interp_y / max_val
            return None

        except Exception as e:
            print(f"Ошибка парсинга спектра: {e}")
            return None

    def parse_xrf_csv(self, filepath):
        """Парсинг CSV с элементным составом"""
        elements_dict = {}
        if not filepath:
            return elements_dict

        try:
            df_xrf = pd.read_csv(filepath, sep=None, engine='python')
            if df_xrf.shape[1] >= 2:
                cols = df_xrf.columns
                for _, row in df_xrf.iterrows():
                    el = str(row[cols[0]]).strip().capitalize()
                    if '_' in el:
                        el = el.split('_')[0]
                    try:
                        elements_dict[el] = float(row[cols[1]])
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Ошибка чтения РФА: {e}")

        return elements_dict

    def parse_and_plot_spectrum(self):
        """Отображение загруженного спектра"""
        if not self.esp_file_path:
            return

        self.target_spectrum = self.parse_esp_file(self.esp_file_path)

        if self.target_spectrum is None:
            messagebox.showerror("Ошибка", "Не удалось прочитать спектр из файла")
            return

        # Очистка и отрисовка
        self.ax1.clear()
        self.ax1.plot(wavenumbers, self.target_spectrum, 'b-', linewidth=1.5, label='Образец')
        self.ax1.set_title("Спектр образца", fontweight='bold')
        self.ax1.set_xlabel("Волновое число (см⁻¹)")
        self.ax1.set_ylabel("Интенсивность (норм.)")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(100, 1500)
        self.ax1.set_ylim(0, 1.1)
        self.ax1.legend()

        self.ax2.clear()
        self.ax2.set_title("Сравнение с базами данных", fontweight='bold')
        self.ax2.set_xlabel("Волновое число (см⁻¹)")
        self.ax2.set_ylabel("Интенсивность (норм.)")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlim(100, 1500)
        self.ax2.set_ylim(0, 1.1)

        self.canvas.draw()

    def analyze_sample(self):
        """Основной анализ образца"""
        # Проверки
        if not self.esp_file_path:
            messagebox.showwarning("Предупреждение", "Загрузите файл спектра")
            return

        if self.target_spectrum is None:
            self.target_spectrum = self.parse_esp_file(self.esp_file_path)
            if self.target_spectrum is None:
                messagebox.showerror("Ошибка", "Не удалось прочитать спектр")
                return

        if len(global_database) == 0:
            messagebox.showwarning("Предупреждение", "Базы данных еще не загружены")
            return

        # Парсинг РФА если есть
        xrf_data = {}
        if self.xrf_file_path:
            xrf_data = self.parse_xrf_csv(self.xrf_file_path)

        # Поиск совпадений
        search_results = []
        for sample_id, ref_spectrum in global_database.items():
            sim = 1 - cosine(self.target_spectrum, ref_spectrum)
            if np.isnan(sim):
                sim = 0.0
            search_results.append((sample_id, sim))

        search_results.sort(key=lambda x: x[1], reverse=True)
        top_matches = search_results[:5]  # Показываем топ-5

        # Отображение результатов
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Топ совпадений:\n\n")

        comp_list = []
        for sample_id, s in top_matches:
            if s > 0.3:
                mineral_name = sample_id.split('__')[0].replace('_', ' ').title()
                rruff_id = sample_id.split('__')[1] if '__' in sample_id else "Unknown"
                result_str = f"{mineral_name}\nID: {rruff_id}\nСходство: {s * 100:.1f}%\n\n"
                self.results_text.insert(tk.END, result_str)
                comp_list.append((mineral_name, rruff_id, s))

        # Построение графика сравнения
        self.ax2.clear()
        self.ax2.plot(wavenumbers, self.target_spectrum, 'b-', linewidth=2, label='Образец', alpha=0.8)

        colors = ['r', 'g', 'm', 'c', 'orange']
        for i, (sample_id, s) in enumerate(top_matches[:3]):
            if s > 0.3:
                ref = global_database[sample_id]
                mineral_name = sample_id.split('__')[0].replace('_', ' ').title()
                self.ax2.plot(wavenumbers, ref, colors[i % len(colors)],
                              linewidth=1, alpha=0.7, linestyle='--',
                              label=f"{mineral_name} ({s * 100:.1f}%)")

        self.ax2.set_title("Сравнение с базами данных", fontweight='bold')
        self.ax2.set_xlabel("Волновое число (см⁻¹)")
        self.ax2.set_ylabel("Интенсивность (норм.)")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlim(100, 1500)
        self.ax2.set_ylim(0, 1.1)
        self.ax2.legend(loc='upper right', fontsize=8)
        self.canvas.draw()

        # Формирование отчета
        self.generate_report(comp_list, xrf_data)

    def generate_report(self, comp_list, xrf_data):
        """Генерация текстового отчета"""
        report = "ЛАБОРАТОРНЫЙ РАПОРТ НТЦ\n"
        report += "=" * 50 + "\n\n"

        # Состав
        report += "Состав:\n"
        if comp_list:
            for mineral_name, rruff_id, s in comp_list[:3]:
                report += f"  • {mineral_name} (Образец {rruff_id}, достоверность {s * 100:.1f}%)\n"
        else:
            report += "  • Специфическая аморфная или органическая матрица\n"
            report += "    (в базах неорганики точных совпадений нет)\n"

        # Примеси
        report += "\nПримеси:\n"
        imp_list = []
        if xrf_data:
            heavy_metals = ["V", "Ni", "Cu", "Zn", "Pb", "Ba", "Sr", "U", "Th", "Ti", "Mn"]
            for el, val in xrf_data.items():
                if el in heavy_metals and val > 0.01:
                    imp_list.append(f"Ионы {el} ({val}%)")
                elif el in ["S", "Cl", "P", "F"] and val > 1.0:
                    imp_list.append(f"Агрессивный анион {el} ({val}%)")

        if imp_list:
            for imp in imp_list:
                report += f"  • {imp}\n"
        elif xrf_data:
            report += "  • Аномальных микропримесей не зафиксировано\n"
        else:
            report += "  • Данные РФА не загружены\n"

        # Технологическое заключение
        report += "\nТехнологическое заключение:\n"
        report_text = ' '.join([item[0] for item in comp_list])

        if "Halite" in report_text or "Sylvite" in report_text or \
                any(el == "Cl" for el in xrf_data.keys()):
            report += "  • Зафиксированы соли/галогениды. Риск электрохимической\n"
            report += "    коррозии и солеотложения. Требуется ингибирование.\n"
        elif "Gypsum" in report_text or "Anhydrite" in report_text or \
                any(el == "S" for el in xrf_data.keys()):
            report += "  • Сульфатная агрессия. Риск образования осадка в трубах.\n"
            report += "    Рекомендуется использование фосфонатов.\n"
        elif any(el in ["V", "Ni"] for el in xrf_data.keys()):
            report += "  • Маркеры тяжелой нефти. Высокая вероятность АСПО.\n"
            report += "    Требуется термообработка.\n"
        elif any(mineral in report_text for mineral in ["Kaolinite", "Illite", "Montmorillonite"]):
            report += "  • Присутствуют глинистые минералы. Риск набухания пласта.\n"
            report += "    Использовать ингибированные буровые растворы (KCl).\n"
        else:
            report += "  • Матрица стабильна. Химически агрессивных осложнений\n"
            report += "    для скважинного оборудования не прогнозируется.\n"

        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(1.0, report)


def main():
    root = tk.Tk()
    app = RamanAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()