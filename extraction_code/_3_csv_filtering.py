import pandas as pd
import os
import shutil
from _0_helper_functions import FlightDataHelper
import json


class CSVFiltering:
    def __init__(self, csv_file_path=None):
        self.base_dir = csv_file_path

    def read_json_data(self, json_path):
        """Read JSON data into a DataFrame."""
        with open(json_path, "r") as f:
            return pd.DataFrame(json.load(f))

    def process_directory(self, dir_path, image_dir):
        good_trajectory_path = os.path.join(image_dir, "good_trajectories")
        bad_trajectory_path = os.path.join(image_dir, "bad_trajectories")

        if os.path.exists(good_trajectory_path) and os.path.exists(bad_trajectory_path):
            return
        os.makedirs(good_trajectory_path, exist_ok=True)
        os.makedirs(bad_trajectory_path, exist_ok=True)

        json_data_path = os.path.join(
            dir_path, "trajectory_report_data", "flight_data.json"
        )
        json_data = self.read_json_data(json_data_path)

        for file in os.listdir(dir_path):
            if not file.endswith(".csv"):
                continue

            # ! Here we are rejecting the file which had error while extraction from ZIP.
            if file[:-4] not in json_data["file_name"].values:
                print("CSV file not in JSON data")
                continue

            flight_csv_path = os.path.join(dir_path, file)

            try:
                df = pd.read_csv(flight_csv_path, on_bad_lines="skip", low_memory=False)
            except Exception as e:
                print(f"Failed to read {flight_csv_path}: {e}")
                continue

            df_ned = FlightDataHelper.gps_to_ned(df)
            within_50, outside_3000 = FlightDataHelper.filter_csv(df_ned)

            plot_path = os.path.join(image_dir, file.replace(".csv", ".png"))

            if not os.path.exists(plot_path):
                print(f"Plot not found for {flight_csv_path}, skipping.")
                continue

            if not within_50 and not outside_3000:
                msg = f"Good Trajectory: {flight_csv_path}"
                print(msg)

                new_plot_path = os.path.join(
                    good_trajectory_path, file.replace(".csv", ".png")
                )
                shutil.move(plot_path, new_plot_path)
                print("Moved plot: Good Trajectory")
            else:
                msg = f"Bad Trajectory: {flight_csv_path}"
                print(msg)

                new_plot_path = os.path.join(
                    bad_trajectory_path, file.replace(".csv", ".png")
                )
                shutil.move(plot_path, new_plot_path)
                print("Moved plot: Bad Trajectory")

    # Main Function
    def main(self):
        print("Starting CSV Filtering...")
        for dirs in os.listdir(self.base_dir):
            dir_path = os.path.join(self.base_dir, dirs)
            if not os.path.isdir(dir_path):
                continue

            trajectory_report_data_path = os.path.join(
                dir_path, "trajectory_report_data"
            )
            if not os.path.isdir(trajectory_report_data_path):
                continue

            image_dir = os.path.join(trajectory_report_data_path, "images")
            if not os.path.isdir(image_dir):
                continue

            print(f"Processing Directory: {dirs}")
            self.process_directory(dir_path, image_dir)
            print(f"Finished processing: {dirs}")
            print("=" * 80)


# # Run the main function if this script is executed directly
# if __name__ == "__main__":
#     csv_file_path = "/home/arjav.singh/Projects/DeepIO/Production_Data/SIDM/csvs"
#     csv_filtering = CSVFiltering(csv_file_path)
#     csv_filtering.main()
