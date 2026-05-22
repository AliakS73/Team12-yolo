"""
Agregacja wyników: z results.csv robi czytelne tabele i wykresy.

Tworzy w results/:
  - summary_table.csv     — uporządkowana tabela zbiorcza
  - plot_map_vs_time.png  — mAP vs średni czas/obraz (kompromis)
  - plot_map_bar.png      — mAP per model (słupki)

Uruchom po zebraniu wyników:
    python src/aggregate_results.py
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # zapis do pliku, bez okna
import matplotlib.pyplot as plt

from config import ensure_results_dir


def main():
    out_dir = ensure_results_dir()
    csv_path = out_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Brak {csv_path}. Najpierw uruchom evaluate.py dla swoich modeli."
        )

    df = pd.read_csv(csv_path)
    df = df.sort_values(["dataset", "model", "imgsz", "iou"]).reset_index(drop=True)

    # krótka etykieta zbioru (results.csv może mieć kilka datasetów: VOC, COCO-20)
    ds_tag = {"VOC_cocomap.yaml": "VOC", "coco20.yaml": "COCO20"}
    df["ds"] = df["dataset"].map(lambda d: ds_tag.get(d, str(d).replace(".yaml", "")))

    # tabela zbiorcza (z kolumną zbioru — inaczej wiersze różnych datasetów są nierozróżnialne)
    summary = df[[
        "ds", "model", "imgsz", "iou", "params_M", "GFLOPs",
        "mAP50", "mAP50_95", "total_time_s", "ms_per_image",
    ]].rename(columns={"ds": "dataset"})
    summary_path = out_dir / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[agg] Tabela zbiorcza: {summary_path}")
    print(summary.to_string(index=False))

    # wykres 1: kompromis mAP vs czas (każdy punkt = jedno uruchomienie)
    plt.figure(figsize=(8, 6))
    for _, r in df.iterrows():
        plt.scatter(r["ms_per_image"], r["mAP50_95"], s=80)
        plt.annotate(f"{r['model']} [{r['ds']}]\n@{int(r['imgsz'])} iou{r['iou']}",
                     (r["ms_per_image"], r["mAP50_95"]),
                     fontsize=7, xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Średni czas na obraz [ms]  (większy = wolniej)")
    plt.ylabel("mAP@50-95  (większy = dokładniej)")
    plt.title("Kompromis dokładność ↔ szybkość (zero-shot: VOC + COCO-20)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = out_dir / "plot_map_vs_time.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"[agg] Wykres kompromisu: {p1}")

    # wykres 2: słupki mAP per uruchomienie
    plt.figure(figsize=(10, 5))
    labels = (df["model"] + " [" + df["ds"] + "] @" + df["imgsz"].astype(str)
              + " iou" + df["iou"].astype(str))
    plt.bar(labels, df["mAP50_95"])
    plt.ylabel("mAP@50-95")
    plt.title("mAP@50-95 dla porównywanych konfiguracji")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    p2 = out_dir / "plot_map_bar.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"[agg] Wykres słupkowy: {p2}")


if __name__ == "__main__":
    main()
