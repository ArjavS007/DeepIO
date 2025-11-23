# Optimized CSV Extraction and Validation with Parallel Processing and Progress Bars

import os
import zipfile
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class ExtractCsvFiles:
    def __init__(self, base_dir=None, target_dir=None):
        self.base_dir = base_dir
        self.target_dir = target_dir

        if self.target_dir is not None:
            os.makedirs(self.target_dir, exist_ok=True)

        # Constants
        self.input_columns = [
            "Gp",
            "Gq",
            "Gr",
            "Ax",
            "Ay",
            "Az",
            "Bx",
            "By",
            "Bz",
            "Altitude",
            "Mode",
        ]
        self.output_columns = ["GPS Lat", "GPS Lon", "GPS AGL"]
        self.date_time_columns = ["GPS Date", "GPS Time"]
        self.required_columns = (
            self.input_columns + self.output_columns + self.date_time_columns
        )

        # Expected dtypes
        self.expected_dtypes = {
            "Gp": "int64",
            "Gq": "int64",
            "Gr": "int64",
            "Ax": "int64",
            "Ay": "int64",
            "Az": "int64",
            "Bx": "int64",
            "By": "int64",
            "Bz": "int64",
            "Altitude": "float64",
            "Mode": "object",
            "GPS Lat": "float64",
            "GPS Lon": "float64",
            "GPS AGL": "float64",
            "GPS Date": "int64",
            "GPS Time": "int64",
        }

    # --------------------------------------------------------
    # Extract CSV from ZIP
    # --------------------------------------------------------
    def extract_csv_from_zip(self, zip_path, dest_folder):
        start = time.perf_counter()
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                os.makedirs(dest_folder, exist_ok=True)
                for member in zip_ref.namelist():
                    if member.lower().endswith(".csv"):
                        zip_ref.extract(member, dest_folder)
            success = True
        except Exception as e:
            print(f"[ZIP ERROR] {zip_path}: {e}")
            success = False
        finally:
            end = time.perf_counter()
            return {"success": success, "time_taken": end - start}

    # --------------------------------------------------------
    # Validate CSV
    # --------------------------------------------------------
    def validate_csv(self, file_path):
        try:
            df = pd.read_csv(file_path, on_bad_lines="skip", low_memory=False)
            df.columns = [col.strip() for col in df.columns]

            # Keep ONLY required columns
            df.drop(
                columns=[c for c in df.columns if c not in self.required_columns],
                inplace=True,
            )
            df.to_csv(file_path, index=False)

            # Missing columns
            missing = set(self.required_columns) - set(df.columns)
            if missing:
                return f"Missing columns: {missing}", None

            # Mode OFF check
            if set(df["Mode"].dropna().unique()).issubset({"OFF", "OFF/0/V"}):
                return None, "Drone is in OFF, OFF/0/V mode throughout"

            # AGL flight check
            if df["GPS AGL"].eq(0).all():
                return None, "Drone did not fly (GPS AGL = 0 throughout)"

            # dtype mismatches
            mismatches = []
            for col, expected in self.expected_dtypes.items():
                if col in df.columns and str(df[col].dtype) != expected:
                    mismatches.append((col, str(df[col].dtype), expected))

            if mismatches:
                msg = ", ".join(
                    [f"{c} (found: {a}, expected: {e})" for c, a, e in mismatches]
                )
                return None, f"Data type mismatch: {msg}"

            return None, None

        except Exception as e:
            return f"Failed to read: {e}", None

    # --------------------------------------------------------
    # Process directory with multithreading + progress bars
    # --------------------------------------------------------
    def process_directory(self, folder_name):
        sub_dir = os.path.join(self.base_dir, folder_name)
        out_dir = os.path.join(self.target_dir, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        error_logs = []
        zip_files = []

        # Collect ZIP files
        for root, _, files in os.walk(sub_dir):
            for file in files:
                if file.endswith(".zip"):
                    zip_files.append(os.path.join(root, file))

        # ---------------------------------------------
        # 1. Parallel ZIP Extraction
        # ---------------------------------------------
        time_taken_to_extract = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.extract_csv_from_zip, z, out_dir): z
                for z in zip_files
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Extracting ZIPs"
            ):
                result = future.result()
                time_taken_to_extract += result["time_taken"]

        print(f"Total time taken to extract CSVs = {time_taken_to_extract:.4f} seconds")

        # ---------------------------------------------
        # 2. Validate CSVs
        # ---------------------------------------------
        csv_files = [f for f in os.listdir(out_dir) if f.endswith(".csv")]

        val_start = time.perf_counter()
        for file in tqdm(csv_files, desc="Validating CSVs"):
            file_path = os.path.join(out_dir, file)
            err_missing, err_other = self.validate_csv(file_path)

            if err_other:
                error_logs.append(f"[SKIPPED] {file_path} - {err_other}")
                os.remove(file_path)

            elif err_missing:
                error_logs.append(f"[SKIPPED] {file_path} - {err_missing}")
                os.remove(file_path)

        val_end = time.perf_counter()
        print(f"Time taken to validate all CSVs = {val_end - val_start:.4f} seconds")

        # ---------------------------------------------
        # 3. Save error report
        # ---------------------------------------------
        if error_logs:
            with open(os.path.join(out_dir, "error_log.txt"), "w") as f:
                f.write("\n".join(error_logs))

        print(f"[DONE] {folder_name}")
        print("-" * 40)

    # --------------------------------------------------------
    # Main
    # --------------------------------------------------------
    def main(self):
        folders = [
            f
            for f in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, f))
        ]
        for folder in folders:
            self.process_directory(folder)
