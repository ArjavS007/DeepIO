from tqdm.autonotebook import tqdm
import pandas as pd
import logging
import os
import pymap3d as pm

class ProcessCSV:
    def __init__(self, base_dir):
        self.base_dir = base_dir

        self.mode_mapping = {
            "COLLISION": 0,
            "FS BATT": 1,
            "FS COMM": 2,
            "HOLD/30/F": 3,
            "HOME": 4,
            "HOME/40/F": 5,
            "HOME/40/V": 6,
            "HOVER": 7,
            "LAND": 8,
            "LAND/21/F": 9,
            "LAND/22/F": 10,
            "LAND/25/V": 11,
            "MANUAL E": 12,
            "OFF": 13,
            "OFF/0/V": 14,
            "RPV": 15,
            "STARTUP": 16,
            "STARTUP/99/V": 17,
            "TAKEOFF": 18,
            "TAKEOFF/10/V": 19,
            "TAKEOFF/11/V": 20,
            "TAKEOFF/12/T": 21,
            "TAKEOFF/12/V": 22,
            "TAKEOFF/13/T": 23,
        }
        
        self.logger = logging.getLogger("ProcessCSV")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def gps_to_ned(self, df):
        # ! Make sure to remove the zero GPS coordinates. Find the first non-zero GPS coordinates for reference
        # Use the first non-zero record as the reference lat, lon, and alt
        lat_ref = df["GPS Lat"].iloc[0]
        lon_ref = df["GPS Lon"].iloc[0]
        alt_ref = df["Altitude"].iloc[0]

        # Convert all GPS coordinates to NED using the reference point
        ned_coords = []
        for i, row in df.iterrows():
            lat, lon, alt = row["GPS Lat"], row["GPS Lon"], row["Altitude"]
            # Convert each (lat, lon, alt) to NED using the reference point
            north, east, down = pm.geodetic2ned(
                lat, lon, alt, lat_ref, lon_ref, alt_ref
            )
            ned_coords.append([north, east, -down])

        # Convert to a DataFrame for easier handling
        ned_df = pd.DataFrame(ned_coords, columns=["x", "y", "z"])
        ned_df["group"] = df["group"]
        ned_df["Mode"] = df["Mode"]

        return ned_df

    def add_delta_NED(self, df_ned_data):
        df_ned_data["delta_x"] = df_ned_data["x"].diff()
        df_ned_data.loc[0, "delta_x"] = 0
        df_ned_data["delta_y"] = df_ned_data["y"].diff()
        df_ned_data.loc[0, "delta_y"] = 0
        df_ned_data["delta_z"] = df_ned_data["z"].diff()
        df_ned_data.loc[0, "delta_z"] = 0

    # Custom aggregation function for delta NED
    def get_last_non_zero_or_last(self, group):
        """
        Return last non zero delta NED if exist else all zero delta NED
        """
        # Identify non-zero rows
        non_zero_rows = group[(group != 0).any(axis=1)]

        if not non_zero_rows.empty:
            # If there are non-zero rows, return the last non-zero row
            return non_zero_rows.iloc[-1]
        else:
            # Otherwise, return the last row (which will be zeros)
            return group.iloc[-1]

    def parse_gps_time(self, df):
        # Ensure GPS Date & Time are zero-padded
        df["date"] = df["GPS Date"].astype(str).str.zfill(6)
        df["time"] = df["GPS Time"].astype(str).str.zfill(6)

        # Combine into one string
        df["datetime_str"] = df["date"] + df["time"]

        # Convert to datetime (invalid values become NaT)
        df["datetime"] = pd.to_datetime(
            df["datetime_str"], format="%d%m%y%H%M%S", errors="coerce"
        )

        # Convert to Unix timestamp (seconds since epoch)
        df["timestamp_seconds"] = df["datetime"].astype("int64") // 10**9

        df.drop(
            columns=[
                "GPS Date",
                "GPS Time",
                "date",
                "time",
                "datetime_str",
                "datetime",
            ],
            inplace=True,
        )

    def process(self):
        self.logger.info("Starting dataset preprocessing")

        for i in tqdm(range(len(self.flight_csv_paths)), desc="Flight#", mininterval=5):
            file_path = self.flight_csv_paths[i]

            try:
                self.logger.info(f"Processing file: {file_path}")
                
                input_columns = [
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
                output_columns = ["GPS Lat", "GPS Lon", "GPS AGL"]
                other_columns = ["GPS Date", "GPS Time"]
                relevant_columns = other_columns + input_columns + output_columns

                avg_columns = [
                    "Bx","By","Bz","Altitude","Gp","Gq","Gr",
                    "Ax","Ay","Az","I Vx","I Vy","I Vz"
                ]

                df = pd.read_csv(file_path, on_bad_lines="skip", low_memory=False)

                df_non_zero = df[
                    (df["GPS Lat"] != 0) &
                    (df["GPS Lon"] != 0) &
                    (df["GPS AGL"] != 0)
                ]
                
                if df_non_zero.empty:
                    self.logger.warning(f"No valid GPS data in file: {file_path}")
                    continue
                
                df = df[df_non_zero.index[0]:].reset_index(drop=True)

                self.parse_gps_time(df)

                df["group"] = df["timestamp_seconds"] - df["timestamp_seconds"].iloc[0]
                df["Mode"] = df["Mode"].map(self.mode_mapping)

                df_ned = df.copy(deep=True)

                agg_map = {col: "mean" for col in avg_columns}
                df = df.groupby(["group", "Mode"], as_index=False, sort=False).agg(agg_map)

                df_in = df[input_columns].copy(deep=True)
                df_in["Altitude"] = df_in["Altitude"].diff()
                df_in.loc[df_in.index[0], "Altitude"] = 0

                df_analysis = df[["GPS Speed", "GPS Head", "I Vx", "I Vy", "I Vz"]]
                df_analysis = df_analysis.groupby(["group", "Mode"]).apply(
                    self.get_last_non_zero_or_last,
                    include_groups=False
                )

                df_ned = self.gps_to_ned(df_ned)
                df_ned = df_ned.groupby(["group", "Mode"]).apply(
                    self.get_last_non_zero_or_last,
                    include_groups=False
                )

                self.add_delta_NED(df_ned)
                df_delta_ned = df_ned[["delta_x", "delta_y", "delta_z"]]

                if (
                    not df_in.isna().any().any()
                    and not df_delta_ned.isna().any().any()
                    and not df_analysis.isna().any().any()
                ):
                    final_df = pd.concat(
                        [
                            df_in.reset_index(drop=True),
                            df_analysis.reset_index(drop=True),
                            df_delta_ned.reset_index(drop=True),
                        ],
                        axis=1,
                    )

                    # base, ext = os.path.splitext(file_path)
                    # out_path = base + "_processed.csv"

                    final_df.to_csv(file_path, index=False)

                    self.logger.info(f"Saved processed file → {file_path}")

                else:
                    self.logger.warning(f"NaN detected — skipping file: {file_path}")

            except Exception as e:
                self.logger.exception(f"Failed processing file: {file_path}")
                
    def main(self):
        self.flight_csv_paths = []
        
        for dir in os.listdir(self.base_dir):
            dir_path = os.path.join(self.base_dir, dir)
            if not os.path.isdir(dir_path):
                continue

            trajectory_report_data_path = os.path.join(
                dir_path, "trajectory_report_data"
            )
            if not os.path.isdir(trajectory_report_data_path):
                continue

            good_image_dir = os.path.join(trajectory_report_data_path, "images", "good_trajectories")
            if not os.path.isdir(good_image_dir):
                continue
            
            for files in os.listdir(good_image_dir):
                if files.endswith(".png"):
                    csv_file = files.replace(".png", ".csv")
                    csv_path = os.path.join(dir_path, csv_file)
                    if os.path.exists(csv_path):
                        self.flight_csv_paths.append(csv_path)
                    else:
                        self.logger.warning(f"Missing CSV for image: {files}")

                    
        self.process()
            
            
        