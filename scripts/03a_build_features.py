# ./scripts/031_build_features.py
# Membangun dataset supervised learning dari data CTT dengan melakukan feature engineering dan menghasilkan file CSV untuk interval prediksi 10, 30, dan 60 menit.

import os

from ui.terminal_display import hr, gap, banner, say_info, say_ok
from pipeline.config import load_config
from pipeline.feature_engineering import (
    scan_timestamp_index,
    preload_frames,
    build_interval_dataset,
)

INTERVALS_MINUTES = [10, 30, 60]
LAG_COUNT = 3


def main():
    cfg = load_config()

    banner("FEATURE ENGINEERING - CTT FORECASTING")
    say_info(f"Folder data   : {cfg.FINAL_BASE_DIR}")

    output_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    os.makedirs(output_dir, exist_ok=True)
    say_info(f"Folder output : {output_dir}")
    hr()

    index = scan_timestamp_index(cfg.FINAL_BASE_DIR)
    say_info(f"Total timestamp terindeks: {len(index)}")
    hr()

    frames, lat, lon = preload_frames(index)
    if not frames:
        say_info("Tidak ada frame yang berhasil dimuat, berhenti.")
        return
    hr()

    for interval in INTERVALS_MINUTES:
        gap()
        banner(f"INTERVAL {interval} MENIT")
        df = build_interval_dataset(frames, lat, lon, interval_minutes=interval, lag_count=LAG_COUNT)

        out_path = os.path.join(output_dir, f"features_{interval}min.csv")
        df.to_csv(out_path, index=False)
        say_ok(f"Disimpan: {out_path}  ({os.path.getsize(out_path) / 1024:.1f} KB)")

    gap()
    banner("SELESAI")
    say_info("Lanjut ke Tahap 3: training model (SVR, XGBoost, LightGBM, CatBoost) per interval.")


if __name__ == "__main__":
    main()