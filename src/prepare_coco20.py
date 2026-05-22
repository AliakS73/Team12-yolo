"""
Przygotowanie "COCO-20" — COCO val2017 ograniczone do tych samych 20 klas co VOC.

Cel: trudniejszy odpowiednik VOC do porównania (TE SAME 20 klas, ale TRUDNIEJSZE
zdjęcia — COCO ma więcej małych obiektów, zatłoczenia, przesłonięć). Dzięki temu
porównanie VOC-20 vs COCO-20 izoluje "trudność obrazów" (klasy są identyczne).

Wciąż ZERO-SHOT: 20 klas ⊂ 80 klas COCO, więc modele COCO-pretrained ewaluujemy
bez treningu (tak jak na VOC).

Pobiera TYLKO val2017 (~1 GB) + etykiety boxowe (~250 MB) — NIE cały COCO (~20 GB).
Następnie:
  1. filtruje etykiety val2017: zostawia tylko linie z 20 klasami VOC (indeksy COCO
     bez zmian), resztę 60 klas usuwa z GT — by mAP liczyło się nad tymi 20 klasami;
  2. hardlinkuje obrazy do równoległego katalogu COCO20 (jak w VOC — symlink katalogu
     ultralytics rozwiązałby do oryginalnych, NIEfiltrowanych etykiet);
  3. zapisuje coco20.yaml (nc=80, nazwy COCO, val = przefiltrowany val2017).

Uruchom RAZ:
    python src/prepare_coco20.py
Potem:
    python src/evaluate.py --model yolo11n.pt --dataset coco20.yaml
"""

import os
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils import ASSETS_URL, SETTINGS
from ultralytics.utils.downloads import download

# 20 klas VOC w indeksach COCO — identyczne z config.classes (filtr predykcji).
VOC_COCO_IDS = {0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62}

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_YAML = REPO_ROOT / "coco20.yaml"
SPLIT = "val2017"


def ensure_coco_val(coco_root: Path):
    """Pobiera (raz) etykiety boxowe + obrazy val2017, jeśli brak. Tylko val!"""
    val_images = coco_root / "images" / SPLIT
    val_labels = coco_root / "labels" / SPLIT
    if not val_labels.is_dir():
        print("[prep] Pobieranie etykiet COCO (boxy, ~250 MB)...")
        # rozpakowuje do {datasets}/coco/labels/... + val2017.txt
        download([ASSETS_URL + "/coco2017labels.zip"], dir=coco_root.parent)
    if not val_images.is_dir() or not any(val_images.glob("*.jpg")):
        print("[prep] Pobieranie obrazów val2017 (~1 GB)...")
        download(["http://images.cocodataset.org/zips/val2017.zip"], dir=coco_root / "images")
    if not val_labels.is_dir() or not val_images.is_dir():
        raise FileNotFoundError(f"Brak {val_labels} lub {val_images} po pobraniu.")
    return val_images, val_labels


def filter_labels(src_labels: Path, dst_labels: Path) -> int:
    """Zostawia tylko linie z 20 klasami VOC (indeksy COCO bez zmian). Zwraca liczbę plików."""
    dst_labels.mkdir(parents=True, exist_ok=True)
    n = 0
    for txt in sorted(src_labels.glob("*.txt")):
        out = [ln for ln in txt.read_text().splitlines()
               if ln.split() and int(ln.split()[0]) in VOC_COCO_IDS]
        (dst_labels / txt.name).write_text("\n".join(out) + ("\n" if out else ""))
        n += 1
    return n


def main():
    coco_root = Path(SETTINGS.get("datasets_dir")) / "coco"
    print(f"[prep] COCO root: {coco_root}")
    src_images, src_labels = ensure_coco_val(coco_root)

    out_root = coco_root.parent / "COCO20"
    out_images = out_root / "images" / SPLIT
    out_labels = out_root / "labels" / SPLIT

    # obrazy: HARDLINKI w prawdziwym katalogu (bez kopiowania ~1 GB).
    # NIE symlink katalogu — ultralytics rozwiązałby go do oryginalnych etykiet COCO
    # (80 klas) zamiast przefiltrowanych (20 klas).
    if out_images.is_symlink():
        out_images.unlink()
    out_images.mkdir(parents=True, exist_ok=True)
    n_images = 0
    for img in src_images.glob("*.jpg"):
        dst = out_images / img.name
        if not dst.exists():
            try:
                os.link(img, dst)
            except OSError:
                shutil.copy2(img, dst)
        n_images += 1

    # etykiety: tylko 20 klas VOC
    n_labels = filter_labels(src_labels, out_labels)
    for cache in out_root.glob("labels/*.cache"):
        cache.unlink()

    names = YOLO("yolo11n.pt").names  # 80 nazw COCO (autorytatywne źródło)
    coco20 = {
        "path": str(out_root),
        "train": f"images/{SPLIT}",   # nieużywane (bez treningu); klucz musi istnieć
        "val": f"images/{SPLIT}",
        "test": f"images/{SPLIT}",
        "nc": len(names),
        "names": {int(k): v for k, v in names.items()},
    }
    with open(OUT_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(coco20, f, allow_unicode=True, sort_keys=False)

    print(f"[prep] Obrazy (hardlinki):   {out_images}  ({n_images} szt.)")
    print(f"[prep] Etykiety (20 klas):   {out_labels}  ({n_labels} plików)")
    print(f"[prep] Zapisano dataset YAML: {OUT_YAML}")
    print("[prep] Gotowe. Teraz: python src/evaluate.py --model yolo11n.pt --dataset coco20.yaml")


if __name__ == "__main__":
    main()
