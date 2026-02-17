import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback
import json
from pathlib import Path
import datetime
import shutil
from _0_helper_functions import FlightDataHelper


class BftDataJson:
    def __init__(self, csv_file_path=None):
        self.csv_file_path = csv_file_path

    def get_metrics_json(self, df, ned_df):
        flight_duration_min, flight_duration_sec = (
            FlightDataHelper.return_flight_duration(df)
        )
        data_frequency = len(df) / flight_duration_sec if flight_duration_sec > 0 else 0

        distance_travelled = 0
        if not ned_df.empty:
            for i in range(1, len(ned_df)):
                distance_travelled += np.linalg.norm(
                    ned_df.iloc[i] - ned_df.iloc[i - 1]
                )

        average_speed = (
            distance_travelled / flight_duration_sec if flight_duration_sec > 0 else 0
        )
        average_acceleration = (
            average_speed / flight_duration_sec if flight_duration_sec > 0 else 0
        )
        average_jerk = (
            average_acceleration / flight_duration_sec if flight_duration_sec > 0 else 0
        )

        min_altitude = df["GPS AGL"].min() if not df.empty else 0
        max_altitude = df["GPS AGL"].max() if not df.empty else 0
        altitude_difference = max_altitude - min_altitude

        num_records = len(df)

        return {
            "flight_duration_min": flight_duration_min,
            "flight_duration_sec": flight_duration_sec,
            "data_frequency": round(data_frequency, 3),
            "distance_travelled": round(distance_travelled, 3),
            "average_speed": round(average_speed, 3),
            "average_acceleration": round(average_acceleration, 3),
            "average_jerk": round(average_jerk, 3),
            "min_altitude": round(min_altitude, 3),
            "max_altitude": round(max_altitude, 3),
            "altitude_difference": round(altitude_difference, 3),
            "num_records": num_records,
        }

    def generate_report(self, flight_csv_file_paths, output_dir):
        """
        Processes CSV files to extract metrics and trajectory images.
        Outputs: JSON with metrics + trajectory PNGs.
        """
        flights_data_json = []

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        image_dir = os.path.join(output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        error_txt_path = os.path.join(output_dir, "bft_data_generation_errors.txt")
        if os.path.exists(error_txt_path):
            os.remove(error_txt_path)

        for file_path in tqdm(flight_csv_file_paths, desc="Processing CSVs"):
            try:
                df = pd.read_csv(
                    file_path, on_bad_lines="skip", index_col=False, low_memory=False
                )
                df = df[
                    (df["GPS Lat"] != 0) & (df["GPS Lon"] != 0) & (df["GPS AGL"] != 0)
                ].reset_index(drop=True)

                if df.empty:
                    raise ValueError(
                        "DataFrame is empty after filtering for valid GPS data."
                    )

                file_name = Path(file_path).stem
                ned_df = FlightDataHelper.gps_to_ned(df)
                trajectory_image_file_path = os.path.join(image_dir, file_name + ".png")
                FlightDataHelper.generate_trajectory_image(
                    ned_df, trajectory_image_file_path
                )

                metrics = self.get_metrics_json(df, ned_df)
                metrics["file_name"] = file_name
                metrics["trajectory_image_file_path"] = trajectory_image_file_path
                flights_data_json.append(metrics)

            except Exception as e:
                err_msg = (
                    f"Error processing file {file_path}: {e}\n{traceback.format_exc()}"
                )
                with open(error_txt_path, "a") as error_file:
                    error_file.write(f"{err_msg}\n\n")
                tqdm.write(f"\n{err_msg}\nSkipped file: {file_path}\n")

        json_path = os.path.join(output_dir, "flight_data.json")
        with open(json_path, "w") as json_file:
            json.dump(flights_data_json, json_file, indent=4)
            print(f"\nMetrics JSON saved to {json_path}")

    # Main function
    def main(self):
        directories = []
        for dir in os.listdir(self.csv_file_path):
            dir_path = os.path.join(self.csv_file_path, dir)
            directories.append(dir_path)

        for i, directory in enumerate(directories):
            print(
                f"{datetime.datetime.now()} Processing directory {i + 1}/{len(directories)}: {directory}"
            )
            directory_path = Path(directory)

            output_dir = os.path.join(directory_path, "trajectory_report_data")

            csv_files = [str(f.resolve()) for f in directory_path.glob("*.csv")]

            if csv_files:
                self.generate_report(csv_files, output_dir)
                print(f"{datetime.datetime.now()} Done with {directory_path}")
            else:
                print(
                    f"{datetime.datetime.now()} No CSV files found in {directory_path}. Skipping."
                )
            print("=" * 80)
