# 05_recursive_evaluation.py
# Melakukan evaluasi recursive multi-step forecasting terhadap seluruh model machine learning pada horizon prediksi 3 jam menggunakan metrik MAE.

import os
import joblib
import numpy as np
import pandas as pd

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import load_ar_dataset, chronological_split
from pipeline.recursive_eval import build_ground_truth_lookup, select_start_points, recursive_predict

HORIZON_MINUTES = 180
TEST_FRAC = 0.15
MAX_START_POINTS = 200  # batasi jumlah titik awal biar evaluasi tidak kelamaan


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    models_dir = os.path.join(cfg.PROJECT_ROOT, "models")

    banner("EVALUASI RECURSIVE MULTI-STEP (HORIZON 3 JAM)")

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

        say_info(f"Mencari titik awal valid (butuh {n_steps} langkah ke depan tanpa gap) ...")
        start_points = select_start_points(test_df, interval, lookup, n_steps, MAX_START_POINTS)
        say_info(f"Titik awal valid ditemukan: {len(start_points)}")

        if not start_points:
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

            say_info(f"Evaluasi recursive: {model_name} ({len(start_points)} titik awal x {n_steps} langkah) ...")

            step_errors = {k: [] for k in range(1, n_steps + 1)}
            for sp in start_points:
                for k, err in recursive_predict(model, scaler, sp, interval, n_steps, lookup):
                    step_errors[k].append(err)

            for k in range(1, n_steps + 1):
                errs = step_errors[k]
                if not errs:
                    continue
                all_results.append({
                    "interval_menit": interval,
                    "model": model_name,
                    "langkah_ke": k,
                    "menit_ke_depan": k * interval,
                    "n_sampel": len(errs),
                    "mae": float(np.mean(errs)),
                })

            last_errs = step_errors[n_steps]
            if last_errs:
                say_ok(
                    f"{model_name:10s} | MAE langkah pertama (+{interval}min): "
                    f"{np.mean(step_errors[1]):.4f}K  |  "
                    f"MAE langkah terakhir (+{HORIZON_MINUTES}min): {np.mean(last_errs):.4f}K"
                )
            else:
                say_error(f"{model_name}: tidak ada data valid di langkah terakhir")

    gap()
    banner("RINGKASAN")
    if all_results:
        result_df = pd.DataFrame(all_results)
        out_path = os.path.join(models_dir, "recursive_evaluation.csv")
        result_df.to_csv(out_path, index=False)
        say_ok(f"Disimpan lengkap (semua langkah): {out_path}")
        hr()

        # Ringkasan: MAE di langkah pertama vs langkah terakhir per model/interval
        first_last = []
        for (interval, model_name), group in result_df.groupby(["interval_menit", "model"]):
            first_row = group[group["langkah_ke"] == group["langkah_ke"].min()].iloc[0]
            last_row = group[group["langkah_ke"] == group["langkah_ke"].max()].iloc[0]
            first_last.append({
                "interval_menit": interval,
                "model": model_name,
                "mae_langkah_1": first_row["mae"],
                f"mae_+{HORIZON_MINUTES}min": last_row["mae"],
                "n_sampel_terakhir": last_row["n_sampel"],
            })
        summary_df = pd.DataFrame(first_last).sort_values(["interval_menit", "model"])
        print(summary_df.to_string(index=False))
    else:
        say_error("Tidak ada hasil evaluasi sama sekali.")

    hr()
    say_info("Lanjut ke Tahap 5: pilih model final per interval, lalu bikin script inference + visualisasi 6-panel.")


if __name__ == "__main__":
    main()