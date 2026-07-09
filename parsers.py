# parsers.py
import re
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.interpolate import interp1d

from config import WAVENUMBERS, ATOMIC_MASSES
from models import XRFSpectrum


class FileParser:
    """Парсер файлов образцов"""

    @staticmethod
    def parse_xrf_file(filepath: str) -> Optional[XRFSpectrum]:
        """
        Парсинг файла РФА (x.txt)
        Возвращает XRFSpectrum или None
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            elements = FileParser._parse_elements(content)
            cps, cps_light = FileParser._parse_cps_blocks(content)

            if not elements:
                return None

            return XRFSpectrum(
                elements=elements,
                cps_data=cps,
                cps_light_data=cps_light
            )
        except Exception as e:
            print(f"Ошибка парсинга РФА: {e}")
            return None

    @staticmethod
    def _parse_elements(content: str) -> Dict[str, float]:
        """Извлечение элементного состава из файла"""
        elements = {}

        # Поиск таблицы с элементами
        lines = content.splitlines()
        start = None

        for i, line in enumerate(lines):
            if 'Эл.' in line and 'PPM' in line:
                start = i + 1
                break

        if start is None:
            return elements

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

    @staticmethod
    def _parse_cps_blocks(content: str) -> Tuple[List[float], List[float]]:
        """Извлечение блоков CPS и CPS Light"""
        cps_data = []
        cps_light_data = []

        # Блок CPS
        cps_pattern = r'<<↓↓CPS↓↓>>\s*(.*?)\s*<<↑↑CPS↑↑>>'
        cps_match = re.search(cps_pattern, content, re.DOTALL)
        if cps_match:
            block = cps_match.group(1).strip()
            for token in block.split():
                try:
                    cps_data.append(float(token.replace(',', '.')))
                except ValueError:
                    pass

        # Блок CPS Light
        cps_light_pattern = r'<<↓↓CPS Light↓↓>>\s*(.*?)\s*<<↑↑CPS Light↑↑>>'
        cps_light_match = re.search(cps_light_pattern, content, re.DOTALL)
        if cps_light_match:
            block = cps_light_match.group(1).strip()
            for token in block.split():
                try:
                    cps_light_data.append(float(token.replace(',', '.')))
                except ValueError:
                    pass

        return cps_data, cps_light_data

    @staticmethod
    def parse_raman_file(filepath: str) -> Optional[np.ndarray]:
        """
        Парсинг файла Раман-спектра (r.txt)
        Возвращает нормализованный спектр или None
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            x_data, y_data = [], []

            for line in content.splitlines():
                cleaned = line.strip()
                if not cleaned or cleaned[0].isalpha() or cleaned.startswith('<'):
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

        except Exception as e:
            print(f"Ошибка парсинга Раман-спектра: {e}")
            return None

    @staticmethod
    def parse_element_csv(filepath: str) -> Dict[str, float]:
        """
        Парсинг CSV-файла с элементным составом для Рамана
        """
        elements = {}
        try:
            df = pd.read_csv(filepath, sep=None, engine='python')
            if df.shape[1] >= 2:
                cols = df.columns
                for _, row in df.iterrows():
                    el = str(row[cols[0]]).strip().capitalize()
                    if '_' in el:
                        el = el.split('_')[0]
                    try:
                        elements[el] = float(row[cols[1]])
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Ошибка чтения CSV: {e}")

        return elements

    @staticmethod
    def load_element_files(directory: str) -> Dict[str, Dict[str, float]]:
        """
        Загрузка всех CSV-файлов из папки element
        Возвращает словарь {имя_файла: {элемент: значение}}
        """
        result = {}
        if not os.path.exists(directory):
            return result

        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                elements = FileParser.parse_element_csv(filepath)
                if elements:
                    # Используем имя файла без расширения как ключ
                    key = filename.replace('.csv', '')
                    result[key] = elements

        return result