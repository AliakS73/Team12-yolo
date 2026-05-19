"""
KROK 1 — Fine-tuning modelu na zbiorze INNYM NIŻ COCO.

Dlaczego ten krok jest konieczny:
    Gotowe wagi YOLO (*.pt) są wytrenowane na COCO (80 klas COCO).
    Na zbiorze o innych klasach (np. zwierzęta afrykańskie) taki model
    "nie zna" tych klas. Trzeba go najpierw dotrenować na danych celu,
    a dopiero potem mierzyć mAP.

To etap DROGI (GPU). Robisz go RAZ na każdy wariant modelu, wagi
zapisują się w results/finetune/<nazwa>/weights/best.pt i potem
ewaluujesz je wielokrotnie skryptem evaluate.py (tani, na CPU).

Przykłady:
    # smoke test na CPU (kilka epok, żeby sprawdzić że działa)
    python src/finetune.py --model yolo11n.pt --epochs 3 --device cpu

    # właściwy fine-tuning na GPU (np. w Google Colab)
    python src/finetune.py --model yolo11n.pt --epochs 50 --device 0
    python src/finetune.py --model yolov8n.pt --epochs 50 --device 0
    python src/finetune.py --model yolo11s.pt --epochs 50 --device 0
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from config import load_config, merge_args, ensure_results_dir


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tuning YOLO na zbiorze innym niż COCO")
    p.add_argument("--model", default=None, help="wariant startowy, np. yolo11n.pt")
    p.add_argument("--dataset", default=None, help="yaml zbioru, np. african-wildlife.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default=None, help="'cpu' lub '0' (GPU)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = merge_args(load_config(), args)
    tcfg = cfg["train"]

    model_name = cfg["model"]
    epochs = args.epochs or tcfg["epochs"]
    batch = args.batch or tcfg["batch"]
    # czytelna nazwa runu: np. "yolo11n_ep50_img640"
    run_name = f"{Path(model_name).stem}_ep{epochs}_img{cfg['imgsz']}"
    out_dir = ensure_results_dir("finetune")

    print(f"[finetune] model={model_name}  dataset={cfg['dataset']}")
    print(f"[finetune] epochs={epochs}  batch={batch}  imgsz={cfg['imgsz']}  "
          f"device={cfg['device']}  seed={cfg['seed']}")

    model = YOLO(model_name)
    model.train(
        data=cfg["dataset"],
        epochs=epochs,
        batch=batch,
        imgsz=cfg["imgsz"],
        device=cfg["device"],
        patience=tcfg["patience"],
        seed=cfg["seed"],            # stały seed = powtarzalność
        project=str(out_dir),
        name=run_name,
        exist_ok=True,
    )

    weights = out_dir / run_name / "weights" / "best.pt"
    print(f"\n[finetune] Gotowe. Najlepsze wagi: {weights}")
    print(f"[finetune] Teraz zewaluuj je:  python src/evaluate.py "
          f"--weights {weights}")


if __name__ == "__main__":
    main()
