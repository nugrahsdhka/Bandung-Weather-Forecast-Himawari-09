# 04b_train_direct_models.py
# Fix #2: latih SATU model terpisah per horizon (+10 s/d +90 menit) dari
# dataset dataset/direct/direct_h{h}.csv (hasil 03c). Setiap horizon selalu
# diprediksi LANGSUNG dari observasi asli (fitur base_time), jadi tidak ada
# compounding error sama sekali untuk horizon yang dicakup di sini.

import os
import joblib
import pandas as pd

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import chronological_split, get_feature_target, train_one_model, evaluate

TEST_FRAC = 0.15


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset", "direct")
    models_dir = os.path.join(cfg.PROJECT_ROOT, "models", "direct")
    os.makedirs(models_dir, exist_ok=True)

    banner("TRAINING MODEL DIRECT MULTI-HORIZON (fix #2)")
    say_info(f"Folder dataset : {dataset_dir}")
    say_info(f"Folder model   : {models_dir}")
    hr()

    summary_rows = []

    for h in cfg.DIRECT_HORIZON_STEPS:
        minutes_ahead = h * 10
        gap()
        banner(f"HORIZON +{minutes_ahead} MENIT (step {h})")

        src_path = os.path.join(dataset_dir, f"direct_h{h}.csv")
        if not os.path.exists(src_path):
            say_error(f"File tidak ditemukan: {src_path}, dilewati (jalankan 03c dulu).")
            continue

        df = pd.read_csv(src_path, parse_dates=["base_time", "target_time"])
        train_df, test_df, cutoff_time = chronological_split(df, test_frac=TEST_FRAC)
        say_info(f"Train: {len(train_df)} baris | Test: {len(test_df)} baris | Cutoff: {cutoff_time}")

        X_train, y_train = get_feature_target(train_df)
        X_test, y_test = get_feature_target(test_df)

        h_dir = os.path.join(models_dir, f"h{h}")
        os.makedirs(h_dir, exist_ok=True)

        for model_name in cfg.MODEL_NAMES:
            say_info(f"Melatih {model_name} ...")
            try:
                model, scaler, elapsed = train_one_model(model_name, X_train, y_train)
                metrics = evaluate(model, X_test, y_test, scaler=scaler)

                joblib.dump(model, os.path.join(h_dir, f"{model_name}.joblib"))
                if scaler is not None:
                    joblib.dump(scaler, os.path.join(h_dir, f"{model_name}_scaler.joblib"))

                say_ok(
                    f"{model_name:10s} | {elapsed:6.1f}s | "
                    f"MAE={metrics['mae']:.4f}K  RMSE={metrics['rmse']:.4f}K  R2={metrics['r2']:.4f}"
                )
                summary_rows.append({
                    "horizon_step": h, "menit_ke_depan": minutes_ahead, "model": model_name,
                    "n_train": len(train_df), "n_test": len(test_df),
                    "waktu_training_detik": round(elapsed, 1),
                    "mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"],
                })
            except ImportError as e:
                say_error(f"Library untuk {model_name} belum terinstall: {e}")
            except Exception as e:
                say_error(f"Gagal melatih {model_name}: {e}")

    gap()
    banner("RINGKASAN")
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(models_dir, "direct_training_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(summary_df.to_string(index=False))
        say_ok(f"Ringkasan disimpan: {summary_path}")

        say_info("Perbandingan MAE direct vs recursive (models/recursive_evaluation.csv) "
                  "bisa dicek manual per horizon yang sama, untuk memutuskan mulai step berapa "
                  "recursive masih layak dipakai (lihat fix #6).")
    else:
        say_error("Tidak ada model yang berhasil dilatih.")

    hr()
    say_info("Model direct siap dipakai lewat pipeline.inference.run_direct_then_recursive_forecast.")


if __name__ == "__main__":
    main()
