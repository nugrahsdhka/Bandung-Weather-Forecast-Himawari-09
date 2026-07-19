# ./scripts/pipeline/direct_horizon.py
# Fix #2: bangun dataset untuk pendekatan DIRECT multi-horizon -- setiap
# horizon (+10, +20, ..., +90 menit) punya target yang diambil LANGSUNG dari
# observasi asli pada base_time + h*interval, BUKAN dari rantai prediksi
# recursive. Ini menghilangkan compounding error sepenuhnya untuk horizon
# yang dicakup, dengan konsekuensi butuh satu model terlatih per horizon.
#
# Fitur input (lat, lon, waktu siklikal, lag tbb_13, delta/tren, neighbor,
# anchor) tetap identik dengan FEATURE_COLUMNS di model_training.py -- yang
# beda hanya target_time & target value-nya.

from datetime import timedelta

import pandas as pd

from pipeline.ground_truth import build_ground_truth_lookup, get_actual
from pipeline.model_training import FEATURE_COLUMNS


def build_direct_dataset(df10, horizon_step, base_interval_minutes=10, lookup=None):
    """
    df10: dataset autoregressive 10-menit yang SUDAH melalui load_ar_dataset
    (artinya sudah punya kolom delta/neighbor/anchor lengkap -- lihat
    pipeline/model_training.load_ar_dataset).

    horizon_step: kelipatan base_interval_minutes, mis. horizon_step=3 berarti
    target di base_time + 30 menit.

    Return DataFrame dengan kolom FEATURE_COLUMNS + base_time + pixel_row/col
    + target_time + target_tbb_13 (target langsung, bukan target 1-langkah
    bawaan file features_10min_ar.csv).
    """
    if lookup is None:
        lookup = build_ground_truth_lookup(df10)

    delta = timedelta(minutes=base_interval_minutes * horizon_step)

    rows = []
    for row in df10.itertuples(index=False):
        target_time = row.base_time + delta
        target_val = get_actual(lookup, row.pixel_row, row.pixel_col, target_time)
        if target_val is None:
            continue
        record = {col: getattr(row, col) for col in FEATURE_COLUMNS}
        record["base_time"] = row.base_time
        record["pixel_row"] = row.pixel_row
        record["pixel_col"] = row.pixel_col
        record["target_time"] = target_time
        record["target_tbb_13"] = target_val
        rows.append(record)

    return pd.DataFrame(rows)
