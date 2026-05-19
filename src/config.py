"""
Wczytywanie centralnej konfiguracji z config.yaml oraz scalanie jej
z argumentami przekazanymi z linii poleceń.

Zasada: config.yaml ustala domyślne wartości, CLI je nadpisuje.
Dzięki temu każdy skrypt ma spójne, jedno źródło ustawień.
"""

from pathlib import Path
import yaml

# config.yaml leży w katalogu nadrzędnym względem src/
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Wczytuje config.yaml i zwraca słownik."""
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku konfiguracyjnego: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_args(config: dict, args) -> dict:
    """
    Nadpisuje wartości z config tymi z argparse, ale tylko jeśli
    użytkownik faktycznie podał argument (wartość != None).
    """
    merged = dict(config)
    for key, value in vars(args).items():
        if value is not None:
            merged[key] = value
    return merged


def ensure_results_dir(subdir: str = "") -> Path:
    """Tworzy (jeśli trzeba) i zwraca katalog na wyniki."""
    base = Path(__file__).resolve().parent.parent / "results"
    target = base / subdir if subdir else base
    target.mkdir(parents=True, exist_ok=True)
    return target
