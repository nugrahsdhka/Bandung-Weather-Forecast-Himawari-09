# 03c_build_direct_horizon_features.py
# Fix #2: bangun dataset DIRECT multi-horizon dari features_10min_ar.csv --
# satu file CSV per horizon (+10 s/d +90 menit, lihat cfg.DIRECT_HORIZON_STEPS),
# target diambil langsung dari observasi asli (bukan rantai prediksi
# recursive). Dipakai oleh 04b_train_direct_models.py.
#
# CATATAN: berbeda dengan 03a/03b, script ini TIDAK incremental (selalu
# rebuild penuh dari features_10min_ar.csv) karena datasetnya jauh lebih
# kecil (base 10 menit saja, tidak perlu buka file .nc lagi) dan lebih
# sederhana untuk dijaga konsistensinya antar horizon.

import os

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import load_ar_dataset
from pipeline.ground_truth import build_ground_truth_lookup
from pipeline.direct_horizon import build_direct_dataset


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    out_dir = os.path.join(dataset_dir, "direct")
    os.makedirs(out_dir, exist_ok=True)

    banner("BANGUN DATASET DIRECT MULTI-HORIZON (fix #2)")

    src_path = os.path.join(dataset_dir, "features_10min_ar.csv")
    if not os.path.exists(src_path):
        say_error(f"File tidak ditemukan: {src_path} (jalankan 03a & 03b dulu)")
        return

    say_info(f"Membaca: {src_path}")
    df10 = load_ar_dataset(src_path)
    say_ok(f"Dimuat: {len(df10)} baris")

    say_info("Membangun ground-truth lookup ...")
    lookup = build_ground_truth_lookup(df10)
    hr()

    for h in cfg.DIRECT_HORIZON_STEPS:
        minutes_ahead = h * 10
        gap()
        banner(f"HORIZON +{minutes_ahead} MENIT (step {h})")

        df_direct = build_direct_dataset(df10, h, base_interval_minutes=10, lookup=lookup)
        if df_direct.empty:
            say_error(f"Tidak ada baris valid untuk horizon +{minutes_ahead} menit, dilewati.")
            continue

        out_path = os.path.join(out_dir, f"direct_h{h}.csv")
        df_direct.to_csv(out_path, index=False)
        say_ok(f"Disimpan: {out_path}  ({len(df_direct)} baris, {df_direct.shape[1]} kolom)")

    gap()
    banner("SELESAI")
    say_info("Lanjut ke 04b_train_direct_models.py untuk melatih 1 model per horizon.")


if __name__ == "__main__":
    main()
