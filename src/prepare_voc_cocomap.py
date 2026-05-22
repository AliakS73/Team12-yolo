"""
Przygotowanie VOC do ewaluacji ZERO-SHOT modeli COCO-pretrained.

Problem:
    Model YOLO (yolo11n.pt itd.) ma 80 klas COCO (indeksy 0–79).
    Etykiety VOC mają 20 klas o WŁASNYCH indeksach (0–19).
    Gdybyśmy puścili `model.val(data='VOC.yaml')`, mAP policzyłby się BŁĘDNIE,
    bo indeks "0" znaczy w VOC "aeroplane", a w COCO "person".

Rozwiązanie (minimum kodu, natywne mAP ultralytics):
    1. Przepisujemy etykiety testowe VOC: id VOC (0–19) -> odpowiadające id COCO.
    2. Tworzymy VOC_cocomap.yaml (nc=80, nazwy COCO, val = przepisany test2007).
    3. W evaluate.py ograniczamy predykcje do 20 klas (parametr `classes=...`),
       żeby model nie generował fałszywych detekcji klas spoza VOC.

Uruchom RAZ (po instalacji zależności). Pobierze VOC przy pierwszym razie:
    python src/prepare_voc_cocomap.py
"""

import os
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

# Mapa: indeks klasy VOC (kolejność z VOC.yaml) -> indeks klasy COCO.
#   0 aeroplane->airplane(4)   1 bicycle(1)        2 bird(14)      3 boat(8)
#   4 bottle(39)               5 bus(5)            6 car(2)        7 cat(15)
#   8 chair(56)                9 cow(19)          10 diningtable->dining table(60)
#  11 dog(16)                 12 horse(17)        13 motorbike->motorcycle(3)
#  14 person(0)               15 pottedplant->potted plant(58)    16 sheep(18)
#  17 sofa->couch(57)         18 train(6)         19 tvmonitor->tv(62)
VOC2COCO = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19,
            60, 16, 17, 3, 0, 58, 18, 57, 6, 62]

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_YAML = REPO_ROOT / "VOC_cocomap.yaml"
SPLIT_DIR = "test2007"   # VOC: split testowy (val w VOC.yaml też wskazuje tutaj)


def remap_labels(src_labels: Path, dst_labels: Path) -> int:
    """Kopiuje etykiety, zamieniając indeks klasy VOC -> COCO. Zwraca liczbę plików."""
    dst_labels.mkdir(parents=True, exist_ok=True)
    n = 0
    for txt in sorted(src_labels.glob("*.txt")):
        out_lines = []
        for line in txt.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            parts[0] = str(VOC2COCO[int(parts[0])])
            out_lines.append(" ".join(parts))
        (dst_labels / txt.name).write_text(
            "\n".join(out_lines) + ("\n" if out_lines else "")
        )
        n += 1
    return n


def main():
    print("[prep] Pobieranie / walidacja VOC (przy pierwszym razie ~2.8 GB)...")
    data = check_det_dataset("VOC.yaml")
    voc_root = Path(data["path"])

    src_images = voc_root / "images" / SPLIT_DIR
    src_labels = voc_root / "labels" / SPLIT_DIR
    if not src_images.is_dir() or not src_labels.is_dir():
        raise FileNotFoundError(
            f"Brak {src_images} lub {src_labels}. VOC nie pobrał się poprawnie."
        )

    out_root = voc_root.parent / "VOC_cocomap"
    out_images = out_root / "images" / SPLIT_DIR
    out_labels = out_root / "labels" / SPLIT_DIR

    # obrazy: HARDLINKI w prawdziwym katalogu (bez kopiowania ~1.7 GB).
    # NIE symlink katalogu — ultralytics rozwiązuje symlink do oryginalnego
    # drzewa VOC i czytałby ORYGINALNE etykiety (id VOC) zamiast przepisanych.
    if out_images.is_symlink():
        out_images.unlink()
    out_images.mkdir(parents=True, exist_ok=True)
    n_images = 0
    for img in src_images.glob("*.jpg"):
        dst = out_images / img.name
        if not dst.exists():
            try:
                os.link(img, dst)              # hardlink (ten sam inode)
            except OSError:
                shutil.copy2(img, dst)         # fallback gdy inny filesystem
        n_images += 1

    # etykiety: przepisane indeksy VOC -> COCO
    n_labels = remap_labels(src_labels, out_labels)
    # usuń stary cache etykiet ultralytics, jeśli istnieje
    for cache in out_root.glob("labels/*.cache"):
        cache.unlink()

    # nazwy 80 klas COCO — z gotowego modelu (autorytatywne źródło)
    names = YOLO("yolo11n.pt").names

    cocomap = {
        "path": str(out_root),
        # ultralytics wymaga klucza 'train' — wskazujemy ten sam split testowy
        # (NIE trenujemy, więc nigdy nie jest używany; ten klucz musi tylko istnieć).
        "train": f"images/{SPLIT_DIR}",
        "val": f"images/{SPLIT_DIR}",
        "test": f"images/{SPLIT_DIR}",
        "nc": len(names),
        "names": {int(k): v for k, v in names.items()},
    }
    with open(OUT_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(cocomap, f, allow_unicode=True, sort_keys=False)

    print(f"[prep] Obrazy testowe (hardlinki): {out_images}  ({n_images} szt.)")
    print(f"[prep] Przepisane etykiety:      {out_labels}  ({n_labels} plików)")
    print(f"[prep] Zapisano dataset YAML:    {OUT_YAML}")
    print(f"[prep] Klasy COCO do filtra:     {sorted(set(VOC2COCO))}")
    print("[prep] Gotowe. Teraz: python src/evaluate.py --model yolo11n.pt")


if __name__ == "__main__":
    main()
