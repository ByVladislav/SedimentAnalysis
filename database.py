# database.py
import os
import zipfile
import tempfile
import re
import threading
import pickle
import gc
import numpy as np
from typing import Dict, List, Optional, Callable
from scipy.interpolate import interp1d

from config import WAVENUMBERS, RRUFF_RAMAN_DIR, RRUFF_XRAY_DIR
from models import MineralInfo, RamanSpectrum


class DatabaseLoader:
    """Загрузчик баз данных RRUFF с кэшированием и фоновой загрузкой"""

    CACHE_FILE = "database_cache.pkl"

    def __init__(self):
        self.raman_db: Dict[str, np.ndarray] = {}
        self.xray_db: List[Dict] = []
        self.is_loading = False
        self.progress_callback: Optional[Callable] = None
        self._load_count = 0
        self._total_count = 0
        self._cache_loaded = False
        self._loading_thread = None
        self._stop_loading = False

    def load_all(self, callback: Optional[Callable] = None, force_reload: bool = False) -> None:
        """Загрузка баз данных с использованием кэша"""
        self.progress_callback = callback

        # Пытаемся загрузить из кэша
        if not force_reload and self._load_from_cache():
            self._cache_loaded = True
            self.is_loading = False
            if self.progress_callback:
                self.progress_callback(self._total_count, self._total_count)
            return

        # Если кэша нет или принудительная перезагрузка
        self._start_background_loading()

    def _start_background_loading(self) -> None:
        """Запуск фоновой загрузки баз данных"""
        if self.is_loading:
            return

        self.is_loading = True
        self._stop_loading = False
        self._loading_thread = threading.Thread(target=self._load_databases_background, daemon=True)
        self._loading_thread.start()

    def _load_databases_background(self) -> None:
        """Фоновая загрузка баз данных с периодическими паузами для UI"""
        try:
            raman_zips = self._get_zip_files(RRUFF_RAMAN_DIR)
            xray_zips = self._get_zip_files(RRUFF_XRAY_DIR)

            # Подсчет общего количества
            self._total_count = 0
            for zip_path in raman_zips:
                self._total_count += self._count_files_in_zip(zip_path, '.txt')
            for zip_path in xray_zips:
                self._total_count += self._count_files_in_zip(zip_path, '.txt')

            self._load_count = 0
            self._update_progress()

            # Загрузка Раман-спектров с паузами
            for zip_path in raman_zips:
                if self._stop_loading:
                    break
                self._load_raman_zip_background(zip_path)

            # Загрузка РФА-данных с паузами
            if not self._stop_loading:
                for zip_path in xray_zips:
                    if self._stop_loading:
                        break
                    self._load_xray_zip_background(zip_path)

            # Сохранение в кэш
            if not self._stop_loading:
                self._save_to_cache()

            self.is_loading = False
            self._update_progress()

            # Принудительная сборка мусора
            gc.collect()

        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            self.is_loading = False

    def _load_raman_zip_background(self, zip_path: str) -> None:
        """Загрузка Раман-спектров из ZIP-архива с паузами"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                file_list = [f for f in zf.namelist() if f.endswith('.txt')]

                for i, filename in enumerate(file_list):
                    if self._stop_loading:
                        break

                    sample_id = filename.replace('.txt', '')
                    try:
                        with zf.open(filename) as f:
                            content = f.read().decode('utf-8', errors='ignore')

                        spectrum = self._parse_raman_spectrum(content)
                        if spectrum is not None:
                            self.raman_db[sample_id] = spectrum
                            self._load_count += 1

                            # Обновление прогресса каждые 10 файлов
                            if self._load_count % 10 == 0:
                                self._update_progress()
                                # Небольшая пауза для UI
                                threading.Event().wait(0.001)
                    except Exception:
                        continue
        except Exception:
            pass

    def _load_xray_zip_background(self, zip_path: str) -> None:
        """Загрузка РФА-данных из ZIP-архива с паузами"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                file_list = [f for f in zf.namelist() if f.endswith('.txt')]

                for i, filename in enumerate(file_list):
                    if self._stop_loading:
                        break

                    try:
                        with zf.open(filename) as f:
                            content = f.read().decode('utf-8', errors='ignore')

                        formula = self._extract_formula(content)
                        if formula:
                            comp = self._parse_formula(formula)
                            if comp:
                                mineral_name = os.path.basename(zip_path).replace('.zip', '')
                                self.xray_db.append({
                                    'name': mineral_name,
                                    'formula': formula,
                                    'comp': comp
                                })
                                self._load_count += 1

                                # Обновление прогресса каждые 10 файлов
                                if self._load_count % 10 == 0:
                                    self._update_progress()
                                    threading.Event().wait(0.001)
                    except Exception:
                        continue
        except Exception:
            pass

    def _load_from_cache(self) -> bool:
        """Загрузка баз данных из кэша"""
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.CACHE_FILE)

        if not os.path.exists(cache_path):
            return False

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            if 'raman_db' in cache_data and 'xray_db' in cache_data:
                self.raman_db = cache_data['raman_db']
                self.xray_db = cache_data['xray_db']
                self._total_count = len(self.raman_db) + len(self.xray_db)
                self._load_count = self._total_count

                print(f"✅ Загружено из кэша: {len(self.raman_db)} спектров, {len(self.xray_db)} минералов")
                return True
        except Exception as e:
            print(f"❌ Ошибка загрузки кэша: {e}")
            return False

        return False

    def _save_to_cache(self) -> None:
        """Сохранение баз данных в кэш"""
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.CACHE_FILE)

        try:
            cache_data = {
                'raman_db': self.raman_db,
                'xray_db': self.xray_db,
                'version': '1.0',
                'timestamp': __import__('datetime').datetime.now().isoformat()
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"✅ Кэш сохранен: {cache_path}")
        except Exception as e:
            print(f"❌ Ошибка сохранения кэша: {e}")

    def _get_zip_files(self, directory: str) -> List[str]:
        """Получение списка ZIP-файлов в директории"""
        if not os.path.exists(directory):
            return []
        return [os.path.join(directory, f) for f in os.listdir(directory)
                if f.endswith('.zip')]

    def _count_files_in_zip(self, zip_path: str, extension: str) -> int:
        """Подсчет файлов в ZIP-архиве"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                return sum(1 for name in zf.namelist() if name.endswith(extension))
        except:
            return 0

    def _update_progress(self) -> None:
        """Обновление прогресса"""
        if self.progress_callback:
            self.progress_callback(self._load_count, self._total_count)

    def _parse_raman_spectrum(self, content: str) -> Optional[np.ndarray]:
        """Парсинг Раман-спектра из содержимого файла"""
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

        if len(x_data) < 10 or len(x_data) != len(y_data):
            return None

        x_arr = np.array(x_data, dtype=float)
        y_arr = np.array(y_data, dtype=float)

        sort_idx = np.argsort(x_arr)
        x_arr = x_arr[sort_idx]
        y_arr = y_arr[sort_idx]

        _, unique_indices = np.unique(x_arr, return_index=True)
        x_arr = x_arr[unique_indices]
        y_arr = y_arr[unique_indices]

        if len(x_arr) < 10:
            return None

        f_int = interp1d(x_arr, y_arr, bounds_error=False, fill_value=0.0)
        interpolated_y = f_int(WAVENUMBERS)

        max_val = np.max(interpolated_y)
        if max_val > 0:
            return interpolated_y / max_val
        return None

    def _extract_formula(self, content: str) -> Optional[str]:
        """Извлечение химической формулы из содержимого файла"""
        for line in content.splitlines():
            if re.search(r'(Formula|FORMULA|Химическая формула)\s*[:=]', line):
                parts = re.split(r'[:=]', line, 1)
                if len(parts) > 1:
                    return parts[1].strip()

        first_line = content.splitlines()[0].strip()
        if re.match(r'^[A-Z][a-z]?[0-9]*(?:\s*[A-Z][a-z]?[0-9]*)*$', first_line):
            return first_line

        return None

    def _parse_formula(self, formula_str: str) -> Dict[str, int]:
        """Парсинг химической формулы"""
        pattern = re.compile(r'([A-Z][a-z]?)([0-9]*)')
        matches = pattern.findall(formula_str)
        comp = {}
        for elem, num in matches:
            if elem == 'O':
                continue
            count = int(num) if num else 1
            comp[elem] = comp.get(elem, 0) + count
        return comp

    def get_stats(self) -> Dict[str, int]:
        """Получение статистики загрузки"""
        return {
            'raman_spectra': len(self.raman_db),
            'xray_minerals': len(self.xray_db),
            'total_loaded': self._load_count,
            'total_available': self._total_count,
            'from_cache': self._cache_loaded,
            'is_loading': self.is_loading
        }

    def force_reload(self, callback: Optional[Callable] = None) -> None:
        """Принудительная перезагрузка баз данных"""
        # Останавливаем текущую загрузку
        self._stop_loading = True
        if self._loading_thread and self._loading_thread.is_alive():
            self._loading_thread.join(timeout=2.0)

        # Удаляем кэш
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.CACHE_FILE)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print(f"✅ Кэш удален: {cache_path}")
            except Exception as e:
                print(f"❌ Ошибка удаления кэша: {e}")

        # Очищаем текущие данные
        self.raman_db.clear()
        self.xray_db.clear()
        self._cache_loaded = False
        self._load_count = 0
        self._total_count = 0

        # Принудительная сборка мусора
        gc.collect()

        # Загружаем заново
        self.progress_callback = callback
        self._start_background_loading()

    def cleanup(self) -> None:
        """Очистка ресурсов"""
        self._stop_loading = True
        if self._loading_thread and self._loading_thread.is_alive():
            self._loading_thread.join(timeout=1.0)
        gc.collect()