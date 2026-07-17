# ./scripts/pipeline/incremental_csv.py
# Utilitas bersama untuk menulis CSV secara incremental dengan aman: atomic write,
# fast-path kalau data baru kronologis, guard terhadap file yang korup/setengah-jalan
# (misal akibat proses lama yang diinterupsi sebelum modul ini ada), dan backup
# otomatis saat mode --rebuild dipakai.

import os
import shutil
from datetime import datetime

import pandas as pd

from ui.terminal_display import say_info, say_ok, say_error


class CorruptDatasetError(Exception):
    """Dilempar kalau file dataset lama terindikasi korup/tidak konsisten."""
    pass


def atomic_write_csv(df, out_path):
    """Tulis DataFrame ke CSV secara atomik menggunakan file sementara dan `os.replace()`, sehingga file asli tidak pernah rusak atau setengah tertulis jika proses terhenti."""
    tmp_path = out_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)


def validate_existing_file(path, dedup_cols, sort_col, parse_dates):
    """Validasi dataset agar bebas duplikasi dan tetap terurut, lalu lempar `CorruptDatasetError` jika file korup atau tidak konsisten."""
    if not os.path.exists(path):
        return

    try:
        df = pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        raise CorruptDatasetError(
            f"Gagal membaca '{path}': {e}\n"
            f"  File ini kemungkinan korup/setengah tertulis (mis. proses sempat mati "
            f"di tengah penulisan sebelum atomic write dipakai). "
            f"Backup file ini lalu jalankan ulang dengan --rebuild."
        )

    dup_mask = df.duplicated(subset=dedup_cols, keep=False)
    if dup_mask.any():
        raise CorruptDatasetError(
            f"'{path}' punya {int(dup_mask.sum())} baris duplikat pada kolom {dedup_cols}.\n"
            f"  Ini indikasi file sempat korup/keinterupsi di tengah proses tulis sebelumnya, "
            f"atau diedit manual. Backup file ini, lalu jalankan ulang dengan --rebuild untuk "
            f"membangun ulang dari nol."
        )

    if not df[sort_col].is_monotonic_increasing:
        raise CorruptDatasetError(
            f"'{path}' tidak terurut berdasarkan '{sort_col}'.\n"
            f"  File ini seharusnya selalu terurut (dijaga otomatis oleh skrip ini). Kalau "
            f"tidak terurut, kemungkinan file pernah diedit manual atau ditulis dengan versi "
            f"skrip lama. Backup file ini, lalu jalankan ulang dengan --rebuild."
        )


def append_incremental(df_new, out_path, dedup_cols, sort_cols, sort_col, parse_dates):
    """Tambahkan `df_new` ke `out_path` secara aman dengan append cepat untuk data kronologis atau merge-sort atomik bila diperlukan, lalu kembalikan total baris hasil akhir."""
    file_exists = os.path.exists(out_path)

    if not file_exists:
        df_sorted = df_new.sort_values(sort_cols).reset_index(drop=True)
        atomic_write_csv(df_sorted, out_path)
        return len(df_sorted)

    existing_max = pd.read_csv(out_path, usecols=[sort_col], parse_dates=[sort_col])[sort_col].max()
    new_min = pd.to_datetime(df_new[sort_col]).min()
    chronological = new_min >= existing_max

    if chronological:
        # Fast-path: tidak perlu baca+sort seluruh file lama.
        df_new_sorted = df_new.sort_values(sort_cols).reset_index(drop=True)
        csv_text = df_new_sorted.to_csv(index=False, header=False)
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            f.write(csv_text)

        with open(out_path, "r", encoding="utf-8") as f:
            total_rows = sum(1 for _ in f) - 1  # minus baris header
        return total_rows
    else:
        # Data baru mengisi gap lama (tidak kronologis) -> perlu full re-sort.
        df_old = pd.read_csv(out_path, parse_dates=parse_dates)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.sort_values(sort_cols).reset_index(drop=True)
        atomic_write_csv(df_all, out_path)
        return len(df_all)


def backup_and_clear(path, reason="rebuild"):
    """Untuk mode `--rebuild`, pindahkan file lama ke folder backup bertimestamp sebelum membuat ulang dataset."""
    if not os.path.exists(path):
        return None

    backup_dir = os.path.join(
        os.path.dirname(path), f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, os.path.basename(path))
    shutil.move(path, backup_path)
    say_info(f"[{reason}] File lama di-backup ke: {backup_path}")
    return backup_path