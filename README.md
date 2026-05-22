# TEMAT 13 — Detekcja jednoetapowa: YOLO (2016)

Projekt na przedmiot **Analiza Danych Obrazowych i Multimedialnych** (ADOM).
Artykuł źródłowy: *You Only Look Once* (Redmon i in., 2016, arXiv:1506.02640).
Implementacja: [ultralytics](https://github.com/ultralytics/ultralytics).

## 1. Cel projektu

Zrozumieć i zmierzyć **kompromis dokładność ↔ szybkość** detektora jednoetapowego YOLO
oraz sprawdzić, jak na **innym zbiorze niż COCO** zachowują się różne **wersje** i **rozmiary**
modeli wytrenowanych na COCO.

## 2. Tezy

- **A.** Nowsza wersja YOLO jest lepsza na COCO, ale na innym zbiorze **niekoniecznie**
  (yolov8n → yolo11n → yolo26n, generacje 2023 → 2024 → 2025).
- **B.** Większy model = wyższe mAP, kosztem czasu inferencji (yolo11n / s / m).
- **C.** Hiperparametr **imgsz** (rozdzielczość) wprost steruje kompromisem dokładność/czas.
- **D.** Hiperparametr **iou** (próg NMS) wpływa na mAP / precision / recall.

## 3. Podejście: ewaluacja ZERO-SHOT na VOC (bez treningu)

Używamy **Pascal VOC** (split `test2007`, 4952 obrazy) jako zbioru innego niż COCO.
Wszystkie **20 klas VOC zawiera się w 80 klasach COCO**, więc modele YOLO wytrenowane na
COCO można ewaluować **bez dotrenowania** — to test transferu COCO → VOC.

Aby mAP liczyło się poprawnie, skrypt `src/prepare_voc_cocomap.py`:
1. pobiera VOC (przez ultralytics),
2. **przepisuje etykiety VOC → indeksy COCO** (np. VOC `aeroplane`→COCO `airplane`),
3. tworzy `VOC_cocomap.yaml` (nc=80, nazwy COCO).

W `evaluate.py` predykcje są dodatkowo ograniczane do 20 klas VOC (`classes=...`).

> Dlaczego bez fine-tuningu? Klasy VOC są już „znane" modelowi z COCO — trening byłby zbędny.
> Dla zbioru z **nowymi** klasami (np. african-wildlife) fine-tuning byłby konieczny.

Dla potwierdzenia tezy porównujemy **trzy poziomy trudności**: **VOC-20** (łatwiejsze) → **COCO-20**
(COCO val2017 ograniczone do tych samych 20 klas — trudniejsze zdjęcia, wciąż zero-shot) →
**COCO-80** (oficjalne liczby ultralytics). Im trudniejszy zbiór, tym większy zysk z nowszego/większego
modelu — szczegóły w `notebooks/experiments.ipynb` i `results/REPORT.md`.

## 4. Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate           # Linux/macOS
pip install -U pip
pip install -r requirements.txt
```

Sprawdzenie:
```bash
python -c "import torch, ultralytics; print('CUDA:', torch.cuda.is_available()); ultralytics.checks()"
```

> **GPU/CPU:** projekt to sama inferencja, więc działa na CPU. GPU używaj tylko, gdy wersja
> sterownika NVIDIA pasuje do buildu CUDA PyTorcha (inaczej `torch.cuda.is_available()` = False —
> wtedy ustaw `device: cpu`). Jeśli `python3 -m venv` zgłosi brak `ensurepip`, doinstaluj
> `python3-venv`/`python3-pip` lub użyj `get-pip.py`.

## 5. Przygotowanie danych (raz)

```bash
python src/prepare_voc_cocomap.py     # pobiera VOC (~2.8 GB) i buduje VOC_cocomap.yaml
python src/prepare_coco20.py          # pobiera COCO val2017 (~1.25 GB) -> coco20.yaml (te same 20 klas, trudniejsze zdjęcia)
```

## 6. Uruchomienie

Cały zestaw eksperymentów (zapisuje `results/results.csv`, tabelę i wykresy):
```bash
bash experiments/run_all.sh           # CPU domyślnie; DEVICE=0 dla GPU
```

Pojedyncze komendy:
| Cel | Komenda |
|---|---|
| Jedna ewaluacja | `python src/evaluate.py --model yolo11n.pt` |
| HP1 — rozdzielczość | `python src/evaluate.py --model yolo11n.pt --imgsz 320` |
| HP2 — próg NMS | `python src/evaluate.py --model yolo11n.pt --iou 0.7` |
| Tabele + wykresy | `python src/aggregate_results.py` |
| Szybki test detekcji | `python src/detect.py --source https://ultralytics.com/images/bus.jpg` |
| Demo: 3 rozmiary, opóźnienia | `python src/demo_compare.py --source <obraz>` |
| Notatnik wyników | `jupyter notebook notebooks/experiments.ipynb` |

## 7. Eksperymenty (konfiguracja w `config.yaml`)

| # | Co badamy | Zmienna | Stałe |
|---|---|---|---|
| Baseline | odniesienie | — | yolo11n, imgsz 640, iou 0.6 |
| A | wersja YOLO | yolov8n → yolo11n → yolo26n | rozmiar n, imgsz 640 |
| B | rozmiar modelu | yolo11 n → s → m | wersja 11, imgsz 640 |
| C | HP1: imgsz | 320 / 640 / 960 | yolo11n |
| D | HP2: iou (NMS) | 0.45 / 0.6 / 0.7 | yolo11n, imgsz 640 |

## 8. Mierzone miary

- **mAP@50** i **mAP@50-95** (jakość detekcji),
- **całkowity czas** przetwarzania zbioru testowego i **ms/obraz** (= czas ÷ liczba obrazów),
- **liczba parametrów [M]** i **GFLOPs** (złożoność modelu).

Metodyka czasu: wszystkie modele na tej samej maszynie (CPU), pierwszy przebieg każdego
modelu to rozgrzewka. W raporcie podaj sprzęt.

## 9. Struktura repo

```
├── config.yaml                  # wszystkie ustawienia w jednym miejscu
├── src/
│   ├── config.py                # wczytywanie config + scalanie z CLI
│   ├── prepare_voc_cocomap.py   # pobranie VOC + remap VOC→COCO -> VOC_cocomap.yaml
│   ├── prepare_coco20.py        # COCO val2017 ograniczone do 20 klas VOC -> coco20.yaml
│   ├── evaluate.py              # ewaluacja zero-shot: mAP + czas -> results.csv
│   ├── aggregate_results.py     # tabela zbiorcza + wykresy
│   ├── detect.py                # szybki test detekcji
│   ├── demo_compare.py          # demo: 3 rozmiary, porównanie opóźnień
│   └── demo_realtime.py         # demo real-time
├── experiments/run_all.sh       # cały przebieg jedną komendą
├── notebooks/experiments.ipynb  # notatnik z wynikami (tabele, wykresy, wnioski)
├── notebooks/demo.ipynb         # notatnik demonstracyjny (idea YOLO)
└── results/                     # results.csv, tabele, wykresy (poza gitem)
```

Licencja `ultralytics`: AGPL-3.0 (właściwa dla zastosowań edukacyjnych).
