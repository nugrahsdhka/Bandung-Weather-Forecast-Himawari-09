# ./scripts/pipeline/spatial_features.py
# Fitur tambahan yang "tidak ikut busuk" seiring recursive forecasting:
#   1. Fitur tetangga (spatial context) -- rata-rata & selisih terhadap 4 piksel
#      di sekitarnya (atas/bawah/kiri/kanan) pada grid 5x7.
#   2. Fitur anchor ke observasi asli terakhir -- membantu model "ingat" nilai
#      nyata terakhir sebelum rantai prediksi dimulai, walaupun window
#      [t, tm1, tm2] sudah 100% hasil prediksi di step-step jauh.
#
# Dipakai konsisten di 3 tempat: feature_engineering.py (training),
# recursive_eval.py & inference.py (evaluasi/inference) -- supaya tidak ada
# celah train/inference mismatch.

import numpy as np

N_LAT_DEFAULT = 5
N_LON_DEFAULT = 7

NEIGHBOR_COLUMNS = ["tbb13_neighbor_mean", "tbb13_neighbor_diff"]
ANCHOR_COLUMNS = ["tbb_13_last_real_obs", "minutes_since_last_real_obs"]
EXTRA_COLUMNS = NEIGHBOR_COLUMNS + ANCHOR_COLUMNS


def neighbor_feature_dict(grid_values, pr, pc, n_lat=N_LAT_DEFAULT, n_lon=N_LON_DEFAULT):
    """
    grid_values: dict {(pixel_row, pixel_col): nilai_tbb13} untuk SEMUA piksel
    pada satu timestamp/step yang sama (boleh campuran observasi asli atau
    hasil prediksi step sebelumnya -- yang penting konsisten dgn caller).

    Return dict fitur: rata-rata tetangga (atas/bawah/kiri/kanan yang ada),
    dan selisih nilai piksel ini terhadap rata-rata tetangganya.
    """
    neighbors = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = pr + di, pc + dj
        if 0 <= ni < n_lat and 0 <= nj < n_lon:
            val = grid_values.get((ni, nj))
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                neighbors.append(val)

    center = grid_values.get((pr, pc))
    if neighbors:
        mean_n = float(np.mean(neighbors))
    else:
        mean_n = float(center) if center is not None else 0.0

    diff_n = float(center) - mean_n if center is not None else 0.0

    return {
        "tbb13_neighbor_mean": mean_n,
        "tbb13_neighbor_diff": diff_n,
    }


def anchor_feature_dict(last_real_obs_value, minutes_since_last_real_obs):
    """Fitur jangkar ke observasi asli terakhir yang valid (belum tersentuh compounding error)."""
    return {
        "tbb_13_last_real_obs": float(last_real_obs_value),
        "minutes_since_last_real_obs": float(minutes_since_last_real_obs),
    }
