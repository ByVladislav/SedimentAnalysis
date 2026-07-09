# models.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MineralInfo:
    """Информация о минерале из базы данных"""
    name: str
    formula: str
    strunz: str
    mineral_id: str
    elements: str
    hyperlink: str
    comments: str
    shift: str


@dataclass
class RamanSpectrum:
    """Раман-спектр"""
    sample_id: str
    mineral_name: str
    rruff_id: str
    wavenumbers: np.ndarray
    intensity: np.ndarray
    normalized: bool = False


@dataclass
class XRFSpectrum:
    """РФА-спектр"""
    elements: Dict[str, float]  # элемент -> PPM
    cps_data: List[float]
    cps_light_data: List[float]


@dataclass
class AnalysisResult:
    """Результат анализа"""
    mineral_name: str
    rruff_id: str
    similarity: float  # для Рамана
    composition_score: float  # для РФА
    formula: str
    strunz_class: str
    elements: Dict[str, float]