"""
Demo działania w czasie rzeczywistym z nałożonym licznikiem FPS.

To bezpośrednia ilustracja głównej tezy artykułu z 2016: detekcja jako
jeden przebieg sieci pozwala działać "na żywo".

Źródło:
    --source 0            kamera internetowa (domyślnie)
    --source data/clip.mp4 plik wideo

Przykłady:
    python src/demo_realtime.py
    python src/demo_realtime.py --source data/clip.mp4 --imgsz 320

Sterowanie: klawisz 'q' kończy demo.
"""

import argparse
import time

import cv2
from ultralytics import YOLO

from config import load_config, merge_args


def parse_args():
    p = argparse.ArgumentParser(description="Demo real-time YOLO")
    p.add_argument("--model", default=None)
    p.add_argument("--source", default="0",
                   help="'0' = kamera, lub ścieżka do pliku wideo")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = merge_args(load_config(), args)

    # kamera podawana jako liczba, plik jako string
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(cfg["model"])
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Nie udało się otworzyć źródła: {source}")

    print("[demo] Naciśnij 'q' aby zakończyć.")
    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            frame,
            imgsz=cfg["imgsz"],
            conf=cfg["conf"],
            device=cfg["device"],
            verbose=False,
        )
        annotated = results[0].plot()

        # wygładzony licznik FPS (średnia krocząca)
        now = time.time()
        inst_fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        fps = 0.9 * fps + 0.1 * inst_fps if fps else inst_fps

        cv2.putText(annotated, f"FPS: {fps:5.1f}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("YOLO 2016 - demo real-time", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[demo] Zakończono.")


if __name__ == "__main__":
    main()
