#!/usr/bin/env bash
# ============================================================
#  Pełny przebieg eksperymentów (TEMAT 13)
#
#  Etap DROGI (fine-tuning) -> najlepiej GPU / Google Colab
#  Etap TANI  (ewaluacja)   -> może być CPU
#
#  Uruchom z katalogu głównego repo:
#     bash experiments/run_all.sh
#
#  Najpierw smoke-test (małe epochs) aby sprawdzić, że wszystko działa,
#  potem podmień EPOCHS=50 na właściwy przebieg.
# ============================================================

set -e
cd "$(dirname "$0")/.."

DEVICE=${DEVICE:-cpu}      # ustaw DEVICE=0 jeśli masz GPU:  DEVICE=0 bash experiments/run_all.sh
EPOCHS=${EPOCHS:-3}        # smoke-test = 3. Właściwy: EPOCHS=50 bash experiments/run_all.sh
FT=results/finetune

echo "### Ustawienia: DEVICE=$DEVICE  EPOCHS=$EPOCHS"
echo

echo "================================================================"
echo " ETAP 1 — FINE-TUNING (eksperyment A: wersje, B: rozmiary)"
echo "================================================================"
for M in yolov8n.pt yolo11n.pt yolo11s.pt yolo11m.pt; do
  echo ">>> fine-tuning $M"
  python src/finetune.py --model "$M" --epochs "$EPOCHS" --device "$DEVICE"
done

echo
echo "================================================================"
echo " ETAP 2 — EWALUACJA (mAP + czas inferencji na zbiorze testowym)"
echo "================================================================"
for M in yolov8n yolo11n yolo11s yolo11m; do
  W="$FT/${M}_ep${EPOCHS}_img640/weights/best.pt"
  echo ">>> ewaluacja $W (imgsz 640)"
  python src/evaluate.py --weights "$W" --imgsz 640 --device "$DEVICE"
done

echo
echo "================================================================"
echo " ETAP 3 — HIPERPARAMETR 1: imgsz (sweep BEZ retrenowania)"
echo "================================================================"
W="$FT/yolo11n_ep${EPOCHS}_img640/weights/best.pt"
for SZ in 320 640 960; do
  echo ">>> yolo11n @ imgsz=$SZ"
  python src/evaluate.py --weights "$W" --imgsz "$SZ" --device "$DEVICE"
done

echo
echo "================================================================"
echo " ETAP 4 — AGREGACJA wyników (tabele + wykresy)"
echo "================================================================"
python src/aggregate_results.py

echo
echo "Gotowe. Wyniki, tabele i wykresy w results/."
echo
echo "HIPERPARAMETR 2 (liczba epok) — uruchom osobno na GPU, np.:"
echo "  for E in 20 50 100; do python src/finetune.py --model yolo11n.pt --epochs \$E --device 0; done"
echo "  potem evaluate.py na każdych wagach i ponownie aggregate_results.py"
