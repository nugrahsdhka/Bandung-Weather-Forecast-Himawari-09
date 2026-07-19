# ./scripts/pipeline/time_features.py
# Encoding waktu siklikal -- dipisah dari recursive_eval.py supaya tidak
# circular import dengan inference.py (yang kini juga butuh fungsi ini
# setelah refactor evaluasi joint-grid di fix #1/#4).

import numpy as np


def cyclical_time_features(ts):
    """Encoding siklikal untuk jam-dalam-hari & hari-dalam-tahun (harus identik dengan yang dipakai saat feature engineering)."""
    hour_frac = ts.hour + ts.minute / 60.0
    doy = ts.timetuple().tm_yday
    return {
        "hour_sin": np.sin(2 * np.pi * hour_frac / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour_frac / 24.0),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }
