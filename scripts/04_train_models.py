# ./scripts/04_train_models.py
# Menjalankan pipeline pelatihan model machine learning pada dataset autoregressive, meliputi pembagian data secara kronologis, training, evaluasi, dan penyimpanan model serta metrik performa.

import os
import joblib
import pandas as pd

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import (
    load_ar_dataset,
    chronological_split,
    get_feature_target,
    train_one_model,
    evaluate,
)

INTERVALS_MINUTES = [10, 30, 60]
MODEL_NAMES = ["svr", "xgboost", "lightgbm", "catboost"]
TEST_FRAC = 0.15


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    models_dir = os.path.join(cfg.PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    banner("TRAINING MODEL - CTT FORECASTING (dataset autoregressive)")
    say_info(f"Folder dataset : {dataset_dir}")
    say_info(f"Folder model   : {models_dir}")
    say_info(f"Test fraction  : {TEST_FRAC} (kronologis)")
    hr()

    summary_rows = []

    for interval in INTERVALS_MINUTES:
        gap()
        banner(f"INTERVAL {interval} MENIT")

        src_path = os.path.join(dataset_dir, f"features_{interval}min_ar.csv")
        if not os.path.exists(src_path):
            say_error(f"File tidak ditemukan: {src_path}, dilewati.")
            continue

        df = load_ar_dataset(src_path)
        train_df, test_df, cutoff_time = chronological_split(df, test_frac=TEST_FRAC)
        say_info(
            f"Train: {len(train_df)} baris | Test: {len(test_df)} baris | "
            f"Cutoff waktu: {cutoff_time}"
        )

        X_train, y_train = get_feature_target(train_df)
        X_test, y_test = get_feature_target(test_df)

        interval_dir = os.path.join(models_dir, f"{interval}min")
        os.makedirs(interval_dir, exist_ok=True)

        for model_name in MODEL_NAMES:
            say_info(f"Melatih {model_name} ...")
            try:
                model, scaler, elapsed = train_one_model(model_name, X_train, y_train)
                metrics = evaluate(model, X_test, y_test, scaler=scaler)

                model_path = os.path.join(interval_dir, f"{model_name}.joblib")
                joblib.dump(model, model_path)
                if scaler is not None:
                    joblib.dump(scaler, os.path.join(interval_dir, f"{model_name}_scaler.joblib"))

                say_ok(
                    f"{model_name:10s} | {elapsed:6.1f}s | "
                    f"MAE={metrics['mae']:.4f}K  RMSE={metrics['rmse']:.4f}K  R2={metrics['r2']:.4f}"
                )

                summary_rows.append({
                    "interval_menit": interval,
                    "model": model_name,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "waktu_training_detik": round(elapsed, 1),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                })
            except ImportError as e:
                say_error(f"Library untuk {model_name} belum terinstall: {e}")
            except Exception as e:
                say_error(f"Gagal melatih {model_name}: {e}")

    gap()
    banner("RINGKASAN")
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(models_dir, "training_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(summary_df.to_string(index=False))
        say_ok(f"Ringkasan disimpan: {summary_path}")
    else:
        say_error("Tidak ada model yang berhasil dilatih.")

    hr()
    say_info("Lanjut ke Tahap 4: evaluasi lebih dalam / pemilihan model terbaik per interval.")


if __name__ == "__main__":
    main()