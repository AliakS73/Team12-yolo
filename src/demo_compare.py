"""
DEMO — jedna wersja YOLO w 3 rozmiarach (n / s / m), porównanie opóźnień.

Cel: pokazać "na żywo", że większy model jest dokładniejszy, ale wolniejszy
— czyli kompromis szybkość/dokładność z artykułu 2016.

Działa w 2 trybach:
  1. obraz / URL  -> wypisuje czas inferencji każdego rozmiaru w tabeli
  2. wideo / kamera (--source 0 lub plik) -> okno z 3 modelami i licznikami

Przykłady:
    python src/demo_compare.py
    python src/demo_compare.py --source data/clip.mp4
    python src/demo_compare.py --source 0          # kamera
"""

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from config import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Demo: 3 rozmiary modelu, opóźnienia")
    p.add_argument("--source", default=None,
                   help="obraz/URL, plik wideo, lub '0' (kamera)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def is_video(src) -> bool:
    if isinstance(src, str) and src.isdigit():
        return True
    return isinstance(src, str) and src.lower().endswith((".mp4", ".avi", ".mov"))


def demo_image(models, source, device):
    """Tryb obrazu: zmierz i wypisz czas inferencji każdego rozmiaru."""
    print(f"\n{'model':>14} | {'inferencja[ms]':>14} | {'obiekty':>8}")
    print("-" * 42)
    for name, m in models.items():
        # rozgrzewka
        m.predict(source, device=device, verbose=False)
        t0 = time.perf_counter()
        res = m.predict(source, device=device, verbose=False)
        dt = (time.perf_counter() - t0) * 1000
        print(f"{name:>14} | {dt:>14.1f} | {len(res[0].boxes):>8}")
    print("\nWniosek: większy model = więcej ms na obraz (większe opóźnienie).")


def demo_video(models, source, device):
    """Tryb wideo: 3 okna obok siebie, każdy z licznikiem FPS."""
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć źródła: {source}")

    print("[demo] Naciśnij 'q' aby zakończyć.")
    fps = {k: 0.0 for k in models}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        panels = []
        for name, m in models.items():
            t0 = time.perf_counter()
            res = m.predict(frame, device=device, verbose=False)
            dt = time.perf_counter() - t0
            inst = 1.0 / max(dt, 1e-6)
            fps[name] = 0.9 * fps[name] + 0.1 * inst if fps[name] else inst

            img = res[0].plot()
            cv2.putText(img, f"{name}  {fps[name]:4.1f} FPS", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            panels.append(cv2.resize(img, (480, 360)))

        cv2.imshow("Porownanie rozmiarow YOLO", np.hstack(panels))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    cfg = load_config()
    source = args.source or cfg["demo"]["source"]

    print("[demo] Wczytywanie modeli (pobiorą się przy pierwszym uruchomieniu)...")
    models = {name: YOLO(name) for name in cfg["demo"]["models"]}

    if is_video(source):
        demo_video(models, source, args.device)
    else:
        demo_image(models, source, args.device)


if __name__ == "__main__":
    main()
