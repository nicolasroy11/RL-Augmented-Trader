from typing import List
import duckdb
import os

from db.settings import DUCKDB_ARCHIVES_PATH

def get_valid_duckdb_files(folder_path: str, window_size: int, min_amount_of_windows_in_file: int = 10) -> List[str]:
    
    min_length = window_size * min_amount_of_windows_in_file
    valid_files = []

    for file in os.listdir(folder_path):
        if file.endswith(".duckdb"):
            db_path = os.path.join(folder_path, file)
            con = duckdb.connect(db_path, read_only=True)

            try:
                # Check what tables actually exist
                tables = con.execute("SHOW TABLES").fetchall()
                table_names = [t[0] for t in tables]

                if "ticks" not in table_names:
                    print(f"[WARN] Skipping {file} — no table named 'ticks'. Found: {table_names}")
                    continue

                length = con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]

            except Exception as e:
                print(f"[ERROR] Could not read {file}: {e}")
                continue

            finally:
                con.close()

            if length >= min_length:
                valid_files.append(db_path)
            else:
                print(f"[INFO] Skipping {file} — only {length} rows (< {min_length})")

    return valid_files


def get_all_duckdb_archive_files() -> List[str]:
    
    valid_files = []
    folder_path = DUCKDB_ARCHIVES_PATH
    for file in os.listdir(folder_path):
        if file.endswith(".duckdb"):
            db_path = os.path.join(folder_path, file)
            con = duckdb.connect(db_path, read_only=True)

            try:
                # Check what tables actually exist
                tables = con.execute("SHOW TABLES").fetchall()
                table_names = [t[0] for t in tables]

                if "ticks" not in table_names:
                    print(f"[WARN] Skipping {file} — no table named 'ticks'. Found: {table_names}")
                    continue

                length = con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]

            except Exception as e:
                print(f"[ERROR] Could not read {file}: {e}")
                continue

            finally:
                con.close()
            valid_files.append(db_path)

    return valid_files
