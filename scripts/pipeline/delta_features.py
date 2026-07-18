# ./scripts/pipeline/delta_features.py

TARGET_VAR = "tbb_13"

# Nama kolom delta/tren -- ditambahkan ke FEATURE_COLUMNS di model_training.py
DELTA_COLUMNS = [
    f"{TARGET_VAR}_delta_t",
    f"{TARGET_VAR}_delta_tm1",
    f"{TARGET_VAR}_accel",
]


def compute_delta_dict(t_val, tm1_val, tm2_val, target_var=TARGET_VAR):
    """Hitung delta/tren dari 3 nilai lag mentah -- dipakai saat membangun satu baris fitur (inference)."""
    delta_t = t_val - tm1_val
    delta_tm1 = tm1_val - tm2_val
    accel = delta_t - delta_tm1
    return {
        f"{target_var}_delta_t": delta_t,
        f"{target_var}_delta_tm1": delta_tm1,
        f"{target_var}_accel": accel,
    }


def add_delta_features(df, target_var=TARGET_VAR):
    """
    Tambahkan kolom delta/tren ke DataFrame yang sudah punya kolom
    {target_var}_t, {target_var}_tm1, {target_var}_tm2 (format AR standar
    proyek ini). Dipanggil sekali setelah load dataset penuh (bukan per-baris)
    supaya cepat -- pakai operasi vektor pandas, bukan loop Python.
    """
    col_t, col_tm1, col_tm2 = f"{target_var}_t", f"{target_var}_tm1", f"{target_var}_tm2"
    missing = [c for c in (col_t, col_tm1, col_tm2) if c not in df.columns]
    if missing:
        raise KeyError(
            f"add_delta_features butuh kolom {[col_t, col_tm1, col_tm2]}, "
            f"tapi tidak ditemukan: {missing}"
        )

    df = df.copy()
    df[f"{target_var}_delta_t"] = df[col_t] - df[col_tm1]
    df[f"{target_var}_delta_tm1"] = df[col_tm1] - df[col_tm2]
    df[f"{target_var}_accel"] = df[f"{target_var}_delta_t"] - df[f"{target_var}_delta_tm1"]
    return df