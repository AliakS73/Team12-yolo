#!/usr/bin/env bash
# ============================================================
#  Pełny przebieg eksperymentów ZERO-SHOT (TEMAT 13, VOC)
#
#  Bez treningu — ewaluujemy modele COCO-pretrained na VOC.
#  Sama inferencja: szybkie, działa na GPU (DEVICE=0) lub CPU.
#
#  Uruchom z katalogu głównego repo:
#     bash experiments/run_all.sh
#     DEVICE=cpu bash experiments/run_all.sh
#
#  Punkt bazowy = yolo11n @ imgsz640, iou0.6 (środek obu sweepów),
#  więc sweepy HP1/HP2 pomijają środek, by nie duplikować wierszy.
# ============================================================

set -e
cd "$(dirname "$0")/.."

DEVICE=${DEVICE:-cpu}        # 'cpu' (domyślnie) lub '0' (GPU, jeśli sterownik pasuje)
PY=.venv/bin/python
[ -x "$PY" ] || PY=python

echo "### DEVICE=$DEVICE"
echo

echo "================================================================"
echo " ETAP 0 — przygotowanie zbiorów (raz): VOC->COCO remap + COCO-20"
echo "================================================================"
$PY src/prepare_voc_cocomap.py
$PY src/prepare_coco20.py        # COCO val2017 ograniczone do 20 klas VOC (~1.25 GB)

echo
echo "================================================================"
echo " ETAP A+B — wersje (n: v8/11/26) i rozmiary (yolo11 n/s/m) @ imgsz640 [VOC]"
echo "          + wersje na rozmiarze s (v8s/11s/26s) — kontrola tezy o wersjach"
echo "================================================================"
for M in yolov8n.pt yolo11n.pt yolo26n.pt yolo11s.pt yolo11m.pt yolov8s.pt yolo26s.pt; do
  echo ">>> $M"
  $PY src/evaluate.py --model "$M" --imgsz 640 --device "$DEVICE"
done

echo
echo "================================================================"
echo " ETAP A+B (COCO-20) — te same modele na trudniejszych zdjęciach (te same 20 klas)"
echo "================================================================"
for M in yolov8n.pt yolo11n.pt yolo26n.pt yolo11s.pt yolo11m.pt yolov8s.pt yolo26s.pt; do
  echo ">>> $M @ COCO-20"
  $PY src/evaluate.py --model "$M" --dataset coco20.yaml --imgsz 640 --device "$DEVICE"
done

echo
echo "================================================================"
echo " ETAP C — HP1: rozdzielczość imgsz (yolo11n; 640 = baseline)"
echo "================================================================"
for SZ in 320 960; do
  echo ">>> yolo11n @ imgsz=$SZ"
  $PY src/evaluate.py --model yolo11n.pt --imgsz "$SZ" --device "$DEVICE"
done

echo
echo "================================================================"
echo " ETAP D — HP2: próg IoU dla NMS (yolo11n; 0.6 = baseline)"
echo "================================================================"
for IOU in 0.45 0.7; do
  echo ">>> yolo11n @ iou=$IOU"
  $PY src/evaluate.py --model yolo11n.pt --imgsz 640 --iou "$IOU" --device "$DEVICE"
done

echo
echo "================================================================"
echo " ETAP E — AGREGACJA (tabela + wykresy)"
echo "================================================================"
$PY src/aggregate_results.py

echo
echo "Gotowe. Wyniki, tabela i wykresy w results/."
