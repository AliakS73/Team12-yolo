"""
Ewaluacja ZERO-SHOT modelu COCO-pretrained na VOC.

Mierzy i zapisuje do results/results.csv (jeden wiersz na uruchomienie):
  - mAP@50 i mAP@50-95   (mean Average Precision — główna miara jakości)
  - całkowity czas przetwarzania CAŁEGO zbioru testowego [s]
  - średni czas na 1 obraz [ms]  = całkowity czas / rzeczywista liczba obrazów
  - inference_ms  — czysty czas inferencji/obraz wg ultralytics (kontrola)
  - liczba parametrów [M] i GFLOPs  (miary złożoności modelu)

Bez treningu: ewaluujemy gotowe wagi COCO bezpośrednio. Predykcje są ograniczane
do 20 klas VOC (config: `classes`), a etykiety mają indeksy COCO (skrypt
prepare_voc_cocomap.py) — dzięki temu mAP liczy się poprawnie.

Przykłady:
    python src/evaluate.py --model yolo11n.pt
    python src/evaluate.py --model yolo11n.pt --imgsz 320      # HP1
    python src/evaluate.py --model yolo11n.pt --iou 0.7        # HP2
"""

import argparse
import csv
import time
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import get_num_params, get_flops

from config import load_config, merge_args, ensure_results_dir, CONFIG_PATH

REPO_ROOT = CONFIG_PATH.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VAL_CONF = 0.001  # niski próg pewności = pełna krzywa PR = poprawne mAP
# kolumny identyfikujące wiersz (do podmiany zamiast duplikowania)
KEY_FIELDS = ("model", "dataset", "split", "imgsz", "iou")


def parse_args():
    p = argparse.ArgumentParser(description="Ewaluacja zero-shot: mAP + czas inferencji")
    p.add_argument("--model", default=None, help="gotowe wagi COCO, np. yolo11n.pt")
    p.add_argument("--dataset", default=None, help="yaml zbioru (domyślnie z config)")
    p.add_argument("--imgsz", type=int, default=None, help="rozdzielczość (HP1)")
    p.add_argument("--iou", type=float, default=None, help="próg NMS (HP2)")
    p.add_argument("--device", default=None, help="'cpu' lub '0'")
    p.add_argument("--split", default=None, help="split: val/test")
    return p.parse_args()


def resolve_dataset(name: str) -> str:
    """Zwraca ścieżkę do yaml zbioru, niezależnie od katalogu uruchomienia."""
    p = Path(name)
    if not p.is_absolute():
        cand = REPO_ROOT / name
        if cand.exists():
            return str(cand)
    return name


def count_images(val_paths) -> int:
    """Rzeczywista liczba obrazów splitu (katalog obrazów lub plik-lista .txt)."""
    if isinstance(val_paths, (str, Path)):
        val_paths = [val_paths]
    total = 0
    for p in val_paths:
        p = Path(p)
        if p.is_dir():
            total += sum(1 for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() == ".txt" and p.exists():
            total += sum(1 for line in p.read_text().splitlines() if line.strip())
    return total


def write_row(csv_path: Path, row: dict):
    """Dopisuje wiersz, podmieniając istniejący o tym samym kluczu (bez duplikatów)."""
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if any(str(r.get(k)) != str(row[k]) for k in KEY_FIELDS)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow(row)


def main():
    args = parse_args()
    cfg = merge_args(load_config(), args)
    out_dir = ensure_results_dir()
    csv_path = out_dir / "results.csv"

    model_name = cfg["model"]
    data_path = resolve_dataset(cfg["dataset"])
    split = cfg["split"]

    model = YOLO(model_name)

    # liczba parametrów i GFLOPs — miary złożoności (niezależne od sprzętu)
    n_params, gflops = 0.0, 0.0
    try:
        n_params = round(get_num_params(model.model) / 1e6, 2)
        gflops = round(get_flops(model.model, cfg["imgsz"]), 2)
    except Exception as e:
        print(f"[eval] uwaga: nie udało się policzyć params/GFLOPs: {e}")

    print(f"[eval] model={model_name}  dataset={cfg['dataset']}  split={split}")
    print(f"[eval] imgsz={cfg['imgsz']}  iou={cfg['iou']}  device={cfg['device']}")

    # --- pomiar czasu przetwarzania CAŁEGO zbioru testowego ---
    t0 = time.perf_counter()
    metrics = model.val(
        data=data_path,
        split=split,
        imgsz=cfg["imgsz"],
        iou=cfg["iou"],
        conf=VAL_CONF,
        batch=cfg.get("batch", 8),
        classes=cfg.get("classes"),
        device=cfg["device"],
        verbose=False,
        plots=False,
    )
    total_time_s = time.perf_counter() - t0

    # rzeczywista liczba obrazów -> średni czas/obraz = całość / liczba obrazów
    dd = check_det_dataset(data_path)
    n_images = count_images(dd.get(split) or dd.get("val"))
    ms_per_image = round(total_time_s * 1000 / n_images, 2) if n_images else 0.0
    inference_ms = round(float(metrics.speed.get("inference", 0.0)), 2)

    row = {
        "model": Path(model_name).stem,
        "dataset": cfg["dataset"],
        "split": split,
        "imgsz": cfg["imgsz"],
        "iou": cfg["iou"],
        "device": cfg["device"],
        "params_M": n_params,
        "GFLOPs": gflops,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "total_time_s": round(total_time_s, 2),
        "n_images": n_images,
        "ms_per_image": ms_per_image,
        "inference_ms": inference_ms,
    }

    write_row(csv_path, row)

    print("\n[eval] Wynik:")
    for k, v in row.items():
        print(f"        {k}: {v}")
    print(f"[eval] Zapisano do: {csv_path}")


if __name__ == "__main__":
    main()
