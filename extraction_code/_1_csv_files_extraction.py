import os
import zipfile
import pandas as pd


class ExtractCsvFiles:
    def __init__(self, base_dir=None, target_dir=None):
        self.base_dir = base_dir
        self.target_dir = target_dir

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

    # Functions
    def extract_csv_from_zip(self, zip_path, dest_folder):
        """Extracts CSVs from a zip archive to a destination folder."""
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                os.makedirs(dest_folder, exist_ok=True)
                for member in zip_ref.namelist():
                    if member.lower().endswith(".csv"):
                        zip_ref.extract(member, dest_folder)
                        print(f"Extracted: {member} → {dest_folder}")
        except (FileNotFoundError, zipfile.BadZipFile) as e:
            print(f"[ZIP ERROR] {zip_path}: {e}")
        except Exception as e:
            print(f"[UNKNOWN ERROR] {zip_path}: {e}")

    def validate_csv(self, file_path):
        """
        Validates:
        1. Required columns exist
        2. Drone actually flew (GPS AGL > 0 at least once)
        3. Data types match EXPECTED_DTYPES (reports which columns mismatch)
        4. Drone not in only OFF/OFF/0/V modes
        """
        try:
            # Read once to check columns
            df = pd.read_csv(file_path, on_bad_lines="skip", low_memory=False)
            df.columns = [col.strip() for col in df.columns]
            df.to_csv(file_path, index=False)  # Save cleaned column names

            # 1. Check for missing columns
            missing = set(self.required_columns) - set(df.columns)
            if missing:
                return f"Missing columns: {missing}", None

            # 2. Check if only OFF modes
            if set(df["Mode"].dropna().unique()).issubset({"OFF", "OFF/0/V"}):
                return None, "Drone is in OFF, OFF/0/V mode throughout"

            # 3. Check if drone flew
            if df["GPS AGL"].eq(0).all():
                return None, "Drone did not fly (GPS AGL = 0 throughout)"

            # 4. Check dtypes manually
            mismatched_cols = []
            for col, expected_dtype in self.expected_dtypes.items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    if actual_dtype != expected_dtype:
                        mismatched_cols.append((col, actual_dtype, expected_dtype))

            if mismatched_cols:
                mismatch_msg = ", ".join(
                    [
                        f"{col} (found: {act}, expected: {exp})"
                        for col, act, exp in mismatched_cols
                    ]
                )
                return None, f"Data type mismatch in columns: {mismatch_msg}"

            return None, None  # Passed all checks

        except Exception as e:
            return f"Failed to read: {e}", None

    def process_directory(self, folder_name):
        sub_dir = os.path.join(self.base_dir, folder_name)
        out_dir = os.path.join(self.target_dir, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        error_logs = []

        # 1. Extract CSVs
        for root, _, files in os.walk(sub_dir):
            for file in files:
                if file.endswith(".zip"):
                    self.extract_csv_from_zip(os.path.join(root, file), out_dir)

        # 2. Validate each extracted CSV
        for file in os.listdir(out_dir):
            if not file.endswith(".csv"):
                continue
            file_path = os.path.join(out_dir, file)
            err_missing, err_other = self.validate_csv(file_path)

            if err_other:  # includes did_not_fly + dtype mismatch
                error_logs.append(f"[SKIPPED] {file_path} - {err_other}")
                os.remove(file_path)
            elif err_missing:
                error_logs.append(f"[SKIPPED] {file_path} - {err_missing}")
                os.remove(file_path)

        # 3. Save error report
        if error_logs:
            with open(os.path.join(out_dir, "error_log.txt"), "w") as f:
                for log in error_logs:
                    f.write(log + "\n")

        print(f"[DONE] {folder_name}")
        print("-" * 40)

    # Main function
    def main(self):
        for folder in os.listdir(self.base_dir):
            self.process_directory(folder)
