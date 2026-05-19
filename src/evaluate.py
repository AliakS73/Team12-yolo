"""
KROK 2 — Ewaluacja dotrenowanego modelu na zbiorze testowym.

Mierzy i zapisuje do results/results.csv (jeden wiersz na uruchomienie):
  - mAP@50 i mAP@50-95   (mean Average Precision — główna miara jakości)
  - całkowity czas inferencji na CAŁYM zbiorze testowym [s]
  - średni czas na 1 obraz [ms]  (= całość / liczba obrazów)
  - liczba parametrów [M] i GFLOPs  (miary złożoności modelu)

To etap TANI — działa na CPU, nie trenuje. Można go powtarzać wiele
razy na tych samych wagach z różnym imgsz/iou (sweep hiperparametru).

Przykłady:
    # ewaluacja dotrenowanych wag, bazowy imgsz
    python src/evaluate.py --weights results/finetune/yolo11n_ep50_img640/weights/best.pt

    # ten sam model, inny imgsz (hiperparametr 1 — bez retrenowania!)
    python src/evaluate.py --weights .../best.pt --imgsz 320
    python src/evaluate.py --weights .../best.pt --imgsz 960
"""

import argparse
import csv
import time
from pathlib import Path

from ultralytics import YOLO

from config import load_config, merge_args, ensure_results_dir


def parse_args():
    p = argparse.ArgumentParser(description="Ewaluacja mAP + czas inferencji")
    p.add_argument("--weights", required=True,
                   help="ścieżka do dotrenowanych wag (best.pt)")
    p.add_argument("--dataset", default=None, help="yaml zbioru")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--device", default=None, help="'cpu' lub '0'")
    p.add_argument("--split", default="test",
                   help="split do ewaluacji: test lub val")
    return p.parse_args()


def count_test_images(metrics) -> int:
    """Liczba obrazów w zbiorze testowym (z wyników ultralytics)."""
    try:
        return int(metrics.speed and metrics.seen) if hasattr(metrics, "seen") else 0
    except Exception:
        return 0


def main():
    args = parse_args()
    cfg = merge_args(load_config(), args)
    out_dir = ensure_results_dir()
    csv_path = out_dir / "results.csv"

    model = YOLO(args.weights)

    # liczba parametrów i GFLOPs — miary złożoności (niezależne od sprzętu)
    n_params, gflops = 0.0, 0.0
    try:
        info = model.info(verbose=False)  # (layers, params, grads, flops)
        n_params = round(info[1] / 1e6, 2)
        gflops = round(info[3], 2)
    except Exception:
        pass

    print(f"[eval] weights={args.weights}")
    print(f"[eval] dataset={cfg['dataset']}  split={args.split}  "
          f"imgsz={cfg['imgsz']}  iou={cfg['iou']}  device={cfg['device']}")

    # --- pomiar czasu CAŁEJ ewaluacji na zbiorze testowym ---
    t0 = time.perf_counter()
    metrics = model.val(
        data=cfg["dataset"],
        split=args.split,
        imgsz=cfg["imgsz"],
        iou=cfg["iou"],
        device=cfg["device"],
        seed=cfg["seed"],
        verbose=False,
    )
    total_time_s = time.perf_counter() - t0

    # liczba obrazów: ultralytics zna ją jako metrics.speed-na-obraz,
    # ale najpewniej liczymy z czasów (sum of speed dict * N nie jest dostępne),
    # więc bierzemy liczbę z atrybutu jeśli jest, inaczej z czasu/obraz.
    per_img_ms = sum(metrics.speed.values())  # preprocess+inference+postprocess
    n_images = int(round(total_time_s * 1000 / per_img_ms)) if per_img_ms else 0

    row = {
        "weights": Path(args.weights).parent.parent.name,
        "dataset": cfg["dataset"],
        "split": args.split,
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
        "ms_per_image": round(per_img_ms, 2),
    }

    # dopisz wiersz do wspólnego CSV (nagłówek tylko za pierwszym razem)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print("\n[eval] Wynik:")
    for k, v in row.items():
        print(f"        {k}: {v}")
    print(f"[eval] Dopisano do: {csv_path}")


if __name__ == "__main__":
    main()
