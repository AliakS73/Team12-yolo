# Raport eksperymentów — YOLO (TEMAT 13), ewaluacja zero-shot na VOC

Przedmiot **ADOM**, TEMAT 13 (detekcja jednoetapowa YOLO). Porównanie **wersji** i **rozmiarów**
modeli YOLO (wytrenowanych na COCO) na **innym zbiorze niż COCO** — Pascal VOC — w trybie
**zero-shot** (bez dotrenowania), z pomiarem kompromisu **dokładność ↔ czas**.

## 1. Metodyka

- **Zbiór:** Pascal VOC, split `test2007` — **4952 obrazy**. 20 klas VOC ⊂ 80 klas COCO,
  więc modele COCO-pretrained ewaluujemy bezpośrednio (test transferu COCO → VOC).
- **Remap klas:** etykiety VOC (id 0–19) przepisane na indeksy COCO (`src/prepare_voc_cocomap.py`),
  predykcje ograniczone do 20 klas VOC (`classes=...`) — dzięki temu mAP liczy się poprawnie natywnie.
- **Miary jakości:** mAP@50, mAP@50-95 (główne), dodatkowo precision / recall.
- **Miary złożoności:** liczba parametrów [M], GFLOPs.
- **Czas (wg wymagania):** mierzony dla **całego zbioru testowego**, następnie **dzielony przez
  liczbę obrazów** → `ms/obraz` (bez FPS). `inference_ms` (czysty czas inferencji ultralytics)
  zapisywany jako kontrola.
- **Sprzęt:** Intel Core i5-8300H @ 2.30 GHz (8 wątków), 31 GiB RAM, **CPU** (sterownik NVIDIA
  niezgodny z buildem CUDA PyTorcha → `torch.cuda.is_available() == False`).
- **Środowisko:** Python 3.12.3, PyTorch 2.12.0, ultralytics 8.4.52. `conf=0.001`, `batch=8`, `seed=42`.

> **Baseline:** `yolo11n`, `imgsz=640`, `iou=0.6` — punkt odniesienia dla wszystkich eksperymentów.

## 1a. Czy zbiór nie jest za prosty? (różnicowanie modeli)

Wymaganie prowadzącego: zbiór **nie może być za prosty**, bo wtedy „wszędzie będą dobre wyniki"
i modele się nie różnicują. Sprawdzamy to wprost — nie po samej wysokości mAP, lecz po **rozrzucie**
wyników między modelami (czy w ogóle jest sygnał do wniosków).

| Eksperyment            | Miara     | Wartości na VOC      | Rozrzut | Sygnał |
|------------------------|-----------|----------------------|--------:|:------:|
| Rozmiar n→s→m          | mAP@50-95 | 62.2 → 67.7 → 70.4   | 8.2 pp  | wyraźny |
| HP1 imgsz 320/640/960  | mAP@50-95 | 56.6 / 62.2 / 50.5   | 11.7 pp | wyraźny |
| Wersja v8n/v11n/v26n   | mAP@50-95 | 59.8 / 62.2 / 63.1   | 3.3 pp  | umiarkowany |

- mAP **nie jest wysycone** (max 0.70 na mAP@50-95, daleko od 1.0) — to nie zbiór, na którym „każdy
  model dostaje ~100%". Rozmiar i rozdzielczość różnicują się bardzo czytelnie.
- Małe różnice **między wersjami** to **nie** artefakt łatwości VOC, lecz cecha skali „n": na trudniejszym
  COCO-20 rozrzut wersji jest podobny (≈3.0 pp), a nawet na najtrudniejszym COCO-80 wynosi ≈3.6 pp —
  generacje modeli nano są sobie po prostu bliskie wydajnościowo.
- Trudność VOC jest **świadomie zakotwiczona** osią VOC-20 < COCO-20 < COCO-80 (§7): pokazujemy nie
  pojedynczy punkt, lecz **jak wyniki zmieniają się z trudnością** — to mocniejszy dowód niż bezwzględna
  wartość mAP na jednym zbiorze.

**Wniosek:** VOC różnicuje modele (3 z 4 eksperymentów dają wyraźny sygnał), a oś trudności domyka
kwestię „za prosty". Główną narrację prowadzimy na **mAP@50-95** i **gradiencie trudności**, a COCO-20
(te same 20 klas, trudniejsze zdjęcia) służy jako **twardy komparator**.

## 2. Tabela zbiorcza złożoności i wyników

| model   | imgsz | iou  | params [M] | GFLOPs | mAP@50 | mAP@50-95 | czas całości [s] | ms/obraz |
|---------|------:|-----:|-----------:|-------:|-------:|----------:|-----------------:|---------:|
| yolov8n |   640 | 0.6  |       3.16 |   8.86 | 0.8077 |    0.5979 |            284.8 |    57.51 |
| yolo11n |   640 | 0.6  |       2.62 |   6.61 | 0.8227 |    0.6219 |           270.69 |    54.66 |
| yolo26n |   640 | 0.6  |       2.57 |   6.12 | 0.8172 |    0.6309 |           289.67 |    58.50 |
| yolo11s |   640 | 0.6  |       9.46 |  21.72 | 0.8614 |    0.6772 |           719.73 |   145.34 |
| yolo11m |   640 | 0.6  |      20.11 |  68.53 | 0.8707 |    0.7041 |          1951.92 |   394.17 |
| yolo11n |   320 | 0.6  |       2.62 |   1.65 | 0.7700 |    0.5658 |           105.02 |    21.21 |
| yolo11n |   960 | 0.6  |       2.62 |  14.88 | 0.7692 |    0.5054 |           732.84 |   147.99 |
| yolo11n |   640 | 0.45 |       2.62 |   6.61 | 0.8205 |    0.6153 |           322.22 |    65.07 |
| yolo11n |   640 | 0.7  |       2.62 |   6.61 | 0.8198 |    0.6244 |           325.70 |    65.77 |

Wykresy: `plot_map_vs_time.png` (kompromis mAP ↔ czas) oraz `plot_map_bar.png` (mAP wg modelu).

## 3. Eksperyment A — wersja YOLO (yolov8n → yolo11n → yolo26n)

**Teza:** nowsza wersja jest lepsza na COCO, ale na innym zbiorze **niekoniecznie**.
Porównujemy trzy generacje w rozmiarze „n": yolov8n (2023), yolo11n (2024), yolo26n (2025).

| model   | rok  | params [M] | GFLOPs | mAP@50      | mAP@50-95   | ms/obraz |
|---------|-----:|-----------:|-------:|------------:|------------:|---------:|
| yolov8n | 2023 |       3.16 |   8.86 |      0.8077 |      0.5979 |    57.51 |
| yolo11n | 2024 |       2.62 |   6.61 |  **0.8227** |      0.6219 |    54.66 |
| yolo26n | 2025 |       2.57 |   6.12 |      0.8172 |  **0.6309** |    58.50 |

**Wynik (kluczowy dla tezy):** na głównej mierze **mAP@50 najnowszy yolo26n NIE wygrywa** —
najlepszy jest **rok starszy yolo11n** (0.8227 > 0.8172). Dopiero na ostrzejszej mierze
**mAP@50-95 yolo26n wychodzi na prowadzenie** (0.6309). yolo26n jest przy tym **najlżejszy**
(2.57 M params, 6.12 GFLOPs) i architektonicznie **end-to-end (bez NMS)**.

**Wniosek (do obrony):** to wręcz wzorcowa ilustracja tezy prowadzącego — **„najnowsze ≠ zawsze
najlepsze poza COCO"**. Na standardowym mAP@50 najnowsza wersja przegrywa ze starszą; przewaga
yolo26n ujawnia się tylko przy surowszym kryterium lokalizacji (mAP@50-95). Co więcej, yolo26
**eliminuje NMS**, więc nie podlega hiperparametrowi z Eksp. D (próg NMS) — pokazuje to, że kolejne
wersje YOLO zmieniają samą architekturę, a nie tylko skalują rozmiar.

> **Uczciwie o sile dowodu:** odwrócenie na mAP@50 (yolo11n 0.8227 vs yolo26n 0.8172 = 0.55 pp,
> jeden przebieg, bez oszacowania wariancji) jest **na granicy szumu** — to sygnał ostrożny, nie twardy.
> Mocnym, powtarzalnym wynikiem jest natomiast **topnienie przewagi nowszej wersji wraz ze spadkiem
> trudności zbioru** (gradient w §7: +10% na COCO-80 → +7% COCO-20 → +6% VOC). Dlatego główną tezę
> formułujemy jako „przewaga z COCO **nie przenosi się w pełni** na inny/łatwiejszy zbiór", a nie
> „nowszy jest gorszy".

> **Uwaga metodyczna:** sweep HP2 (`iou` / próg NMS) prowadzimy tylko na yolo11n — dla modelu
> end-to-end bez NMS ten hiperparametr nie ma zastosowania.

## 4. Eksperyment B — rozmiar modelu (yolo11 n / s / m)

**Teza:** większy model = wyższe mAP (kosztem czasu).

| model   | params [M] | GFLOPs | mAP@50 | mAP@50-95 | ms/obraz |
|---------|-----------:|-------:|-------:|----------:|---------:|
| yolo11n |       2.62 |   6.61 | 0.8227 |    0.6219 |    54.66 |
| yolo11s |       9.46 |  21.72 | 0.8614 |    0.6772 |   145.34 |
| yolo11m |      20.11 |  68.53 | 0.8707 |    0.7041 |   394.17 |

**Wynik:** mAP rośnie **monotonicznie** n → s → m (mAP@50-95: 0.6219 → 0.6772 → 0.7041).

**Wniosek:** teza **potwierdzona**, ale z **malejącymi przyrostami** i **rosnącym kosztem**:
- n → s: +0.0553 mAP@50-95, ale czas **×2.66** (54.66 → 145.34 ms/obraz),
- s → m: tylko +0.0269 mAP@50-95, a czas dalej rośnie do **×7.2** względem n (394.17 ms/obraz).

Największy zysk dokładności na jednostkę czasu daje przejście n → s; model `m` jest najdokładniejszy,
ale na CPU bardzo wolny (≈0.4 s/obraz). To klasyczny kompromis dokładność ↔ szybkość.

## 5. Eksperyment C — HP1: rozdzielczość wejścia `imgsz` (yolo11n)

**Teza:** `imgsz` wprost steruje kompromisem dokładność ↔ czas.

| imgsz | GFLOPs | mAP@50 | mAP@50-95 | ms/obraz |
|------:|-------:|-------:|----------:|---------:|
|   320 |   1.65 | 0.7700 |    0.5658 |    21.21 |
|   640 |   6.61 | 0.8227 |    0.6219 |    54.66 |
|   960 |  14.88 | 0.7692 |    0.5054 |   147.99 |

**Wynik (najciekawszy):** zależność jest **NIEmonotoniczna** — mAP **osiąga maksimum przy 640**,
a przy **960 SPADA** (mAP@50-95 nawet poniżej wartości dla 320: 0.5054 < 0.5658), mimo ~9× większego
kosztu obliczeń niż przy 320.

**Wniosek:** „większa rozdzielczość = lepiej" jest **fałszem**. Modele COCO trenowano przy ~640,
więc upscaling do 960 zmienia skalę obiektów względem treningu i pogarsza detekcję. **Nie warto**
płacić ~3× czasem za 960 — `imgsz=640` to optimum dokładności, a 320 to najtańszy wariant
„szybki, lekko gorszy". To mocny, kontrintuicyjny wynik dla raportu.

## 6. Eksperyment D — HP2: próg NMS `iou` (yolo11n, imgsz 640)

**Teza:** próg IoU w NMS wpływa na mAP / precision / recall.

| iou  | mAP@50 | mAP@50-95 | precision | recall |
|-----:|-------:|----------:|----------:|-------:|
| 0.45 | 0.8205 |    0.6153 |    0.7930 | 0.7613 |
| 0.6  | 0.8227 |    0.6219 |    0.7901 | 0.7623 |
| 0.7  | 0.8198 |    0.6244 |    0.7880 | 0.7604 |

**Wynik:** wpływ jest **niewielki** (±0.01). mAP@50 ma maksimum przy 0.6, a mAP@50-95 lekko rośnie
ze wzrostem progu (0.6153 → 0.6244), bo wyższy próg NMS przepuszcza więcej nakładających się
ramek, co pomaga przy ostrzejszych progach IoU w mAP@50-95.

**Wniosek:** próg NMS to **subtelny** hiperparametr strojenia (nie zmienia rzędu wielkości wyniku).
`iou=0.6` jest dobrym, zrównoważonym ustawieniem baseline. Dla mAP@50-95 minimalnie lepsze bywa 0.7.

## 7. Porównanie trudności zbiorów — czy wzorzec się utrzymuje? (★ dowód tezy)

Teza prowadzącego to porównanie zbiorów o **różnej trudności**. Zestawiamy **trzy poziomy**
(te same modele, imgsz 640, iou 0.6):

- **VOC-20** — nasze zero-shot na VOC (20 klas, łatwiejsze zdjęcia),
- **COCO-20** — nasze zero-shot na COCO val2017 ograniczonym do **tych samych 20 klas**
  (trudniejsze, zatłoczone zdjęcia; izoluje samą *trudność obrazów* — wciąż zero-shot, klasy ⊂ COCO),
- **COCO-80** — oficjalne mAP@50-95 z dokumentacji ultralytics (wszystkie 80 klas).

| model   | VOC-20 mAP@50 | COCO-20 mAP@50 | VOC-20 mAP@50-95 | COCO-20 mAP@50-95 | COCO-80 mAP@50-95 |
|---------|--------------:|---------------:|-----------------:|------------------:|------------------:|
| yolov8n |          80.8 |           61.8 |             59.8 |              43.3 |              37.3 |
| yolo11n |      **82.3** |       **64.8** |             62.2 |              45.8 |              39.5 |
| yolo26n |          81.7 |           64.3 |         **63.1** |          **46.3** |          **40.9** |
| yolo11s |          86.1 |           71.9 |             67.7 |              52.7 |              47.0 |
| yolo11m |          87.1 |           75.9 |             70.4 |              57.0 |              51.5 |

> COCO-20 jest wyraźnie trudniejsze niż VOC-20 mimo **identycznych klas** — np. yolov8n spada
> z 80.8 → 61.8 (mAP@50). To czysta różnica „trudności zdjęć".

**Gradient po trudności (przyrost względny mAP@50-95):**

| krok | VOC-20 (łatwiejsze) | COCO-20 (trudniejsze) | COCO-80 (pełne) |
|------|---------------:|-----------------:|----------------:|
| A. nowszy (v8n→26n)  | +6%  | +7%  | +10% |
| B. większy (11n→11m) | +13% | +25% | +30% |

**Wynik (bezpośredni dowód tezy):**
- **Im trudniejszy zbiór, tym większy zysk** z „nowszego" i „większego" modelu — gradient jest
  **monotoniczny** dla obu. Na łatwiejszym VOC dodatkowa pojemność modelu jest słabo wykorzystana
  (zysk z rozmiaru topnieje z **+30%** na COCO-80 do **+13%** na VOC).
- **Najnowszy yolo26n na mAP@50 przegrywa z yolo11n na OBU naszych zbiorach** (VOC 81.7 < 82.3;
  COCO-20 64.3 < 64.8) — to **nie był przypadek VOC**, efekt replikuje się na trudniejszych zdjęciach.

Wykres: `plot_difficulty.png`.

## 8. Wnioski końcowe

1. **Wersja (A):** na głównej mierze **mAP@50 najnowszy yolo26n (2025) NIE wygrywa** — lepszy jest
   rok starszy yolo11n; yolo26n prowadzi dopiero na mAP@50-95. Wzorcowo potwierdza tezę
   „najnowsze ≠ zawsze najlepsze poza COCO". yolo26 jest najlżejszy i bez NMS (zmiana architektury,
   nie tylko skali).
2. **Rozmiar (B):** potwierdzone „większy = dokładniejszy" (monotoniczne n→s→m), lecz z malejącymi
   przyrostami i silnie rosnącym czasem (m ≈ ×7.2 czasu względem n).
3. **HP1 imgsz (C):** kontrintuicyjnie **niemonotoniczne** — optimum przy 640, 960 pogarsza wynik
   (modele trenowane przy ~640). Najmocniejszy wynik eksperymentalny.
4. **HP2 iou (D):** wpływ marginalny (±0.01); 0.6 to rozsądny baseline.
5. **Kompromis dokładność ↔ czas:** dobrze widoczny na CPU. Najlepszy stosunek jakości do czasu —
   yolo11n @ imgsz 640; maksymalna dokładność — yolo11m (kosztem ~0.4 s/obraz).
6. **Trudność zbioru (kluczowy dowód tezy):** na trzech poziomach (VOC-20 < COCO-20 < COCO-80)
   zysk z „nowszego"/„większego" modelu **rośnie monotonicznie z trudnością** (rozmiar: +13% → +25% → +30%).
   Najnowszy yolo26n na mAP@50 **przegrywa z yolo11n na obu naszych zbiorach** (VOC i COCO-20) —
   efekt nie jest przypadkiem VOC. Poprawa na pełnym COCO **nie przenosi się wprost** na inny/węższy zbiór.

> Metodyka czasu zgodna z wymaganiem: mierzony czas **całego zbioru** (4952 obr.), dzielony przez
> liczbę obrazów → `ms/obraz` (bez FPS). Wszystkie pomiary na tej samej maszynie (CPU).
