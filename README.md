## 4. Instalacja 
```bash
python -m venv .venv                 # tworzy izolowane środowisko
# source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate               # Windows (PowerShell)

pip install --upgrade pip
pip install -r requirements.txt
```

4. Sprawdź, że działa:
```bash
python -c "import ultralytics; ultralytics.checks()"
```
Powinno wypisać wersje i `Setup complete`.

> Po każdym ponownym otwarciu VS Code pamiętaj o `source .venv/bin/activate`
> (terminal musi pokazywać `(.venv)` na początku linii).

## 5. Pierwszy test (czy w ogóle działa)
```bash
python src/detect.py --source https://ultralytics.com/images/bus.jpg
```
Jeśli w `results/detect/` pojawi się obrazek z ramkami — środowisko gra.

## 6. Pełny przebieg eksperymentów
**Najpierw smoke-test** (3 epoki, na CPU, żeby sprawdzić że pipeline działa):

```bash
bash experiments/run_all.sh
```

**Potem właściwy przebieg** (50 epok, na GPU — np. w Colab):
```bash
DEVICE=0 EPOCHS=50 bash experiments/run_all.sh
```
Skrypt po kolei: dotrenuje modele → policzy mAP i czas inferencji →
zsweepuje `imgsz` → wygeneruje tabele i wykresy w `results/`.

### Pojedyncze komendy (gdy chcesz coś uruchomić ręcznie)

| Cel | Komenda |
|---|---|
| Fine-tuning jednego modelu | `python src/finetune.py --model yolo11n.pt --epochs 50 --device 0` |
| Ewaluacja (mAP + czas) | `python src/evaluate.py --weights results/finetune/yolo11n_ep50_img640/weights/best.pt` |
| Sweep imgsz (hiperparametr 1) | `python src/evaluate.py --weights .../best.pt --imgsz 320` |
| Tabele i wykresy | `python src/aggregate_results.py` |
| Demo: 3 rozmiary, opóźnienia | `python src/demo_compare.py --source data/clip.mp4` |
| Notebook demonstracyjny | `jupyter notebook notebooks/demo.ipynb` |

## 7. Plan eksperymentów

| # | Co badamy | Zmienna | Stałe | Oczekiwanie |
|---|---|---|---|---|
| Baseline | odniesienie | — | YOLO11n, imgsz 640, 50 ep | — |
| A | wersja YOLO | YOLOv8n → YOLO11n | rozmiar n | nowszy ≠ zawsze lepszy poza COCO |
| B | rozmiar modelu | YOLO11 n → s → m | wersja 11 | większy = wyższy mAP, wolniejszy |
| C | hiperparametr 1: imgsz | 320 / 640 / 960 | YOLO11n | większy imgsz = wyższy mAP, wolniej |
| D | hiperparametr 2: epoki | 20 / 50 / 100 | YOLO11n | więcej epok = lepiej, potem plateau |

Mierzone: **mAP@50, mAP@50-95**, **całkowity czas inferencji na zbiorze
testowym**, **średni czas/obraz**, **liczba parametrów + GFLOPs**.

> Metodyka pomiaru czasu (ważne): wszystkie modele mierz **na tej samej
> maszynie**, zamknij inne programy, laptop podłącz do zasilania, pierwszy
> przebieg każdego modelu to „rozgrzewka" (pomijany). W raporcie podaj,
> na jakim sprzęcie mierzyłeś.

## 8. Zbiór danych

Domyślnie `african-wildlife.yaml` (auto-pobieranie, ~1500 obrazów,
4 klasy: buffalo, elephant, rhino, zebra). Zmienisz go w `config.yaml`
(pole `dataset`). Trudniejsza alternatywa: `VisDrone.yaml` (drony, małe
obiekty — tam teza „nowszy nie zawsze lepszy" widać najmocniej, ale 2,3 GB).

## 9. Struktura repo

```
├── README.md
├── requirements.txt / environment.yml
├── config.yaml                 # wszystkie ustawienia w jednym miejscu
├── src/
│   ├── config.py               # wczytywanie config + CLI
│   ├── detect.py               # szybki test detekcji
│   ├── finetune.py             # ETAP 1: dotrenowanie na zbiorze != COCO
│   ├── evaluate.py             # ETAP 2: mAP + czas inferencji -> results.csv
│   ├── aggregate_results.py    # ETAP 3: tabele + wykresy
│   ├── demo_compare.py         # demo: 3 rozmiary, opóźnienia
│   └── demo_realtime.py        # demo real-time z licznikiem FPS
├── notebooks/demo.ipynb        # notebook demonstracyjny
├── experiments/run_all.sh      # pełny przebieg jedną komendą
├── data/                       # dane (poza gitem)
└── results/                    # wyniki, CSV, wykresy (poza gitem)
```

## 10. Podział pracy

| Rola | Pliki |
|---|---|
| Theory Lead | prezentacje (analiza artykułu) |
| **Implementation Lead (Ty)** | **całe repo, środowisko, pipeline** |
| Experiments Lead | `experiments/`, `results/`, analiza tabel/wykresów |
| Presentation/Demo Lead | `notebooks/demo.ipynb`, `demo_compare.py` |

Licencja: `ultralytics` na AGPL-3.0 (właściwa dla zastosowań edukacyjnych).
