# ./scripts/pipeline/incremental_csv.py
# Utilitas bersama untuk menulis CSV secara incremental dengan aman: atomic write, fast-path kalau data baru kronologis, guard terhadap file yang korup/setengah-jalan (misal akibat proses lama yang diinterupsi sebelum modul ini ada), dan backup otomatis saat mode --rebuild dipakai.

import os
import shutil
from datetime import datetime

import pandas as pd

from ui.terminal_display import say_info, say_ok, say_error


class CorruptDatasetError(Exception):
    """Dilempar kalau file dataset lama terindikasi korup/tidak konsisten."""
    pass


def atomic_write_csv(df, out_path):
    """
    Tulis DataFrame ke CSV secara atomik: tulis dulu ke file sementara di folder yang
    sama, baru di-rename ke nama aslinya. `os.replace()` di Python itu atomik baik di
    POSIX maupun Windows (beda dengan os.rename yg di Windows bisa gagal kalau file
    tujuan sudah ada) -- jadi kalau proses diinterupsi (Ctrl+C, mati listrik, crash)
    persis saat menulis, file ASLI tidak akan pernah dalam kondisi setengah tertulis:
    yang ada cuma file .tmp yang gagal/tidak lengkap, sedangkan file asli tetap versi
    lama yang utuh.
    """
    tmp_path = out_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)


def validate_existing_file(path, dedup_cols, sort_col, parse_dates):
    """
    Guard terhadap file lama yang korup/tidak konsisten -- misal karena run lama
    (sebelum atomic write dipakai) sempat diinterupsi di tengah penulisan, atau
    file diedit manual. Dua hal yang dicek:
      1. Tidak ada baris duplikat pada `dedup_cols` (mis. base_time+pixel_row+pixel_col).
      2. Kolom `sort_col` (base_time) terurut naik -- ini jadi prasyarat supaya
         fast-path kronologis di `append_incremental` aman dipakai.
    Kalau salah satu gagal, lempar CorruptDatasetError dengan pesan yang jelas
    (bukan diam-diam lanjut memproses data yang berpotensi tidak konsisten).
    """
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
    """
    Tambahkan `df_new` ke `out_path` dengan aman:
      - File belum ada          -> tulis baru (terurut), atomik.
      - Data baru kronologis    -> FAST-PATH: langsung append di akhir file tanpa
                                    baca-ulang+sort seluruh isi file (murah & cepat).
                                    "Kronologis" berarti sort_col minimum di df_new >=
                                    sort_col maksimum yang sudah ada di file.
      - Data baru TIDAK kronologis (mis. mengisi gap lama) -> baca ulang seluruh file,
                                    gabung, sort ulang, tulis atomik (lebih mahal, tapi
                                    hasil tetap terurut rapi).
    Return: total baris di file setelah operasi ini.
    """
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
        # Tulis baris baru ke buffer teks sekali jalan, lalu satu kali write() ke file
        # dalam mode append -- meminimalkan jumlah operasi tulis terpisah supaya jendela
        # risiko interupsi (Ctrl+C dsb) menulis baris rusak jadi sekecil mungkin.
        # (Ini bukan atomik 100% seperti rewrite penuh, tapi risikonya cuma baris
        # terakhir yang berpotensi rusak -- dan itu akan langsung ketahuan lewat
        # validate_existing_file() di run berikutnya, bukan korupsi diam-diam.)
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
    """
    Untuk mode --rebuild: JANGAN hapus file lama begitu saja -- pindahkan (backup) dulu
    ke folder `_backup_YYYYMMDD_HHMMSS/` di direktori yang sama, supaya kalau --rebuild
    dipanggil tidak sengaja, data lama tidak hilang permanen.
    """
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