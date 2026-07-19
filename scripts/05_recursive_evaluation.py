# 05_recursive_evaluation.py
# Melakukan evaluasi recursive multi-step forecasting terhadap seluruh model
# machine learning pada horizon prediksi 3 jam. Sejak fix #1, evaluasi
# dilakukan joint (semua piksel bersamaan per titik awal t0) sehingga bisa
# menghitung metrik anti-collapse (spatial_collapse_ratio, spatial_correlation)
# selain MAE -- MAE saja bisa "tertipu" oleh model yang kolaps ke rata-rata.

import os
import joblib
import numpy as np
import pandas as pd

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import load_ar_dataset, chronological_split
from pipeline.ground_truth import build_ground_truth_lookup
from pipeline.recursive_eval import (
    select_valid_t0, evaluate_recursive_at_t0,
    spatial_collapse_ratio, spatial_correlation,
)

HORIZON_MINUTES = 180
TEST_FRAC = 0.15
MAX_START_POINTS = 40  # jumlah titik awal t0 (masing2 mencakup SEMUA piksel sekaligus)

# Ambang batas collapse ratio: di bawah ini model dianggap "kolaps ke rata-rata"
# dan sebisa mungkin tidak dipilih sebagai model terbaik walau MAE-nya rendah.
COLLAPSE_RATIO_THRESHOLD = 0.5


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    models_dir = os.path.join(cfg.PROJECT_ROOT, "models")

    banner("EVALUASI RECURSIVE MULTI-STEP (HORIZON 3 JAM) - joint-grid + anti-collapse")

    say_info("Membangun ground-truth lookup dari features_10min_ar.csv (resolusi native) ...")
    df10 = load_ar_dataset(os.path.join(dataset_dir, "features_10min_ar.csv"))
    lookup = build_ground_truth_lookup(df10)
    say_ok(f"Lookup selesai: {len(lookup)} entri")
    hr()

    all_results = []

    for interval in cfg.INTERVALS_MINUTES:
        gap()
        banner(f"INTERVAL {interval} MENIT")

        n_steps = HORIZON_MINUTES // interval
        src_path = os.path.join(dataset_dir, f"features_{interval}min_ar.csv")
        if not os.path.exists(src_path):
            say_error(f"File tidak ditemukan: {src_path}, dilewati.")
            continue

        df = load_ar_dataset(src_path)
        train_df, test_df, cutoff_time = chronological_split(df, test_frac=TEST_FRAC)
        say_info(f"Cutoff waktu (sama seperti Tahap 3): {cutoff_time}")

        say_info(f"Mencari titik awal (t0) valid (butuh {n_steps} langkah ke depan tanpa gap, semua piksel) ...")
        t0_list = select_valid_t0(test_df, interval, lookup, n_steps, MAX_START_POINTS)
        say_info(f"Titik awal (t0) valid ditemukan: {len(t0_list)}")

        if not t0_list:
            say_error("Tidak ada titik awal valid untuk interval ini, dilewati.")
            continue

        interval_dir = os.path.join(models_dir, f"{interval}min")

        for model_name in cfg.MODEL_NAMES:
            model_path = os.path.join(interval_dir, f"{model_name}.joblib")
            if not os.path.exists(model_path):
                say_error(f"Model tidak ditemukan: {model_path}, dilewati.")
                continue

            model = joblib.load(model_path)
            scaler_path = os.path.join(interval_dir, f"{model_name}_scaler.joblib")
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

            say_info(f"Evaluasi recursive: {model_name} ({len(t0_list)} titik awal x {n_steps} langkah, joint-grid) ...")

            # Kumpulkan preds/actuals per langkah, digabung dari SEMUA t0.
            step_preds = {k: [] for k in range(1, n_steps + 1)}
            step_actuals = {k: [] for k in range(1, n_steps + 1)}

            for t0 in t0_list:
                per_step = evaluate_recursive_at_t0(model, scaler, test_df, t0, interval, n_steps, lookup)
                for entry in per_step:
                    step_preds[entry["langkah_ke"]].extend(entry["preds"].tolist())
                    step_actuals[entry["langkah_ke"]].extend(entry["actuals"].tolist())

            for k in range(1, n_steps + 1):
                preds = step_preds[k]
                actuals = step_actuals[k]
                if not preds:
                    continue
                mae = float(np.mean(np.abs(np.array(preds) - np.array(actuals))))
                collapse_ratio = spatial_collapse_ratio(preds, actuals)
                corr = spatial_correlation(preds, actuals)
                all_results.append({
                    "interval_menit": interval,
                    "model": model_name,
                    "langkah_ke": k,
                    "menit_ke_depan": k * interval,
                    "n_sampel": len(preds),
                    "mae": mae,
                    "collapse_ratio": collapse_ratio,
                    "spatial_corr": corr,
                })

            last_k = n_steps
            last_rows = [r for r in all_results if r["interval_menit"] == interval and r["model"] == model_name and r["langkah_ke"] == last_k]
            first_rows = [r for r in all_results if r["interval_menit"] == interval and r["model"] == model_name and r["langkah_ke"] == 1]
            if last_rows and first_rows:
                say_ok(
                    f"{model_name:10s} | MAE +{interval}min: {first_rows[0]['mae']:.4f}K "
                    f"(collapse={first_rows[0]['collapse_ratio']:.2f})  |  "
                    f"MAE +{HORIZON_MINUTES}min: {last_rows[0]['mae']:.4f}K "
                    f"(collapse={last_rows[0]['collapse_ratio']:.2f}, corr={last_rows[0]['spatial_corr']:.2f})"
                )
            else:
                say_error(f"{model_name}: tidak ada data valid di langkah terakhir")

    gap()
    banner("RINGKASAN")
    if all_results:
        result_df = pd.DataFrame(all_results)
        out_path = os.path.join(models_dir, "recursive_evaluation.csv")
        result_df.to_csv(out_path, index=False)
        say_ok(f"Disimpan lengkap (semua langkah, termasuk collapse_ratio & spatial_corr): {out_path}")
        hr()

        first_last = []
        for (interval, model_name), group in result_df.groupby(["interval_menit", "model"]):
            first_row = group[group["langkah_ke"] == group["langkah_ke"].min()].iloc[0]
            last_row = group[group["langkah_ke"] == group["langkah_ke"].max()].iloc[0]
            collapsed_flag = "KOLAPS" if last_row["collapse_ratio"] < COLLAPSE_RATIO_THRESHOLD else "ok"
            first_last.append({
                "interval_menit": interval,
                "model": model_name,
                "mae_langkah_1": first_row["mae"],
                f"mae_+{HORIZON_MINUTES}min": last_row["mae"],
                f"collapse_ratio_+{HORIZON_MINUTES}min": last_row["collapse_ratio"],
                "status": collapsed_flag,
                "n_sampel_terakhir": last_row["n_sampel"],
            })
        summary_df = pd.DataFrame(first_last).sort_values(["interval_menit", "model"])
        print(summary_df.to_string(index=False))
    else:
        say_error("Tidak ada hasil evaluasi sama sekali.")

    hr()
    say_info("Lanjut ke Tahap 6: inference (select_best_model kini mempertimbangkan collapse_ratio, lihat pipeline/inference.py).")


if __name__ == "__main__":
    main()
