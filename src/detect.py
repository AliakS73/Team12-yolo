import argparse
from ultralytics import YOLO

from config import load_config, merge_args, ensure_results_dir


def parse_args():
    p = argparse.ArgumentParser(description="Detekcja YOLO na obrazie/wideo")
    p.add_argument("--model", default=None, help="ścieżka/nazwa modelu (.pt)")
    p.add_argument("--source", default=None, help="obraz, katalog, wideo lub URL")
    p.add_argument("--imgsz", type=int, default=None, help="rozdzielczość wejścia")
    p.add_argument("--conf", type=float, default=None, help="próg pewności")
    p.add_argument("--iou", type=float, default=None, help="próg IoU dla NMS")
    p.add_argument("--device", default=None, help="'cpu' lub '0' (GPU)")
    return p.parse_args()


def main():
    cfg = merge_args(load_config(), parse_args())
    out_dir = ensure_results_dir("detect")

    print(f"[detect] model={cfg['model']}  source={cfg['source']}  "
          f"imgsz={cfg['imgsz']}  device={cfg['device']}")

    model = YOLO(cfg["model"])
    results = model.predict(
        source=cfg["source"],
        imgsz=cfg["imgsz"],
        conf=cfg["conf"],
        iou=cfg["iou"],
        device=cfg["device"],
        save=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
    )

    total = sum(len(r.boxes) for r in results)
    print(f"[detect] Wykryto łącznie {total} obiektów.")
    print(f"[detect] Wyniki zapisane w: {out_dir}")


if __name__ == "__main__":
    main()
