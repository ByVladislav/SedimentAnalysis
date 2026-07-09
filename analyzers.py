# analyzers.py
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cosine

from config import ATOMIC_MASSES
from models import AnalysisResult


class Analyzer:
    """Анализатор спектров"""

    @staticmethod
    def compare_raman_spectra(target: np.ndarray, database: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
        """
        Сравнение Раман-спектра с базой данных
        Возвращает список (sample_id, similarity)
        """
        results = []

        # Используем локальные переменные для скорости
        target_flat = target.flatten()

        for sample_id, ref_spectrum in database.items():
            try:
                # Косинусное расстояние
                sim = 1 - cosine(target_flat, ref_spectrum.flatten())
                if np.isnan(sim):
                    sim = 0.0
                results.append((sample_id, sim))
            except Exception:
                results.append((sample_id, 0.0))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def compare_raman_spectra_batch(target: np.ndarray, database: Dict[str, np.ndarray],
                                    batch_size: int = 1000) -> List[Tuple[str, float]]:
        """
        Пакетное сравнение Раман-спектра с базой данных
        Оптимизировано для больших баз данных
        """
        results = []
        items = list(database.items())
        target_flat = target.flatten()

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            for sample_id, ref_spectrum in batch:
                try:
                    sim = 1 - cosine(target_flat, ref_spectrum.flatten())
                    if np.isnan(sim):
                        sim = 0.0
                    results.append((sample_id, sim))
                except Exception:
                    results.append((sample_id, 0.0))

            # Периодическая очистка памяти
            if i % (batch_size * 10) == 0:
                gc.collect()

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def compare_composition(sample_elements: Dict[str, float],
                            mineral_comp: Dict[str, int]) -> float:
        """
        Сравнение элементного состава образца с минералом
        Возвращает евклидово расстояние (чем меньше, тем лучше)
        """
        # Пересчет PPM в мольные доли
        sample_moles = {}
        for elem, ppm in sample_elements.items():
            if elem in ATOMIC_MASSES:
                sample_moles[elem] = ppm / ATOMIC_MASSES[elem]

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

        # Евклидово расстояние
        diff = 0.0
        for elem in common:
            diff += (sample_norm[elem] - mineral_norm[elem]) ** 2
        for elem in set(sample_norm.keys()) - set(mineral_norm.keys()):
            diff += sample_norm[elem] ** 2
        for elem in set(mineral_norm.keys()) - set(sample_norm.keys()):
            diff += mineral_norm[elem] ** 2

        return diff

    @staticmethod
    def find_best_xray_match(sample_elements: Dict[str, float],
                             xray_db: List[Dict]) -> Optional[Tuple[Dict, float]]:
        """
        Поиск наилучшего совпадения по РФА
        Возвращает (минерал, score) или None
        """
        if not sample_elements or not xray_db:
            return None

        best = None
        best_score = float('inf')

        for mineral in xray_db:
            score = Analyzer.compare_composition(sample_elements, mineral['comp'])
            if score < best_score:
                best_score = score
                best = mineral

        return (best, best_score) if best else None

    @staticmethod
    def combine_results(raman_matches: List[Tuple[str, float]],
                        xrf_result: Optional[Tuple[Dict, float]],
                        element_data: Dict[str, float]) -> Dict:
        """
        Объединение результатов Рамана и РФА
        """
        result = {
            'raman_matches': raman_matches[:5],
            'xrf_match': xrf_result,
            'elements': element_data,
            'combined_mineral': None
        }

        if xrf_result and raman_matches:
            xrf_name = xrf_result[0]['name'].lower()
            for sample_id, sim in raman_matches[:10]:
                if xrf_name in sample_id.lower():
                    result['combined_mineral'] = {
                        'name': sample_id.split('__')[0].replace('_', ' ').title(),
                        'raman_similarity': sim,
                        'xrf_score': xrf_result[1],
                        'rruff_id': sample_id.split('__')[1] if '__' in sample_id else 'Unknown'
                    }
                    break

        return result