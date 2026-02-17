import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from _0_helper_functions import FlightDataHelper


class Plots:
    def __init__(self, csv_file_path=None):
        self.base_dir = csv_file_path

        # Colors for charts
        self.colors = {
            "good": "#4CAF50",
            "bad": "#D43A3A",
            "duration": "skyblue",
            "distance": "#1E2586",
            "x": "green",
            "y": "orange",
            "z": "purple",
            "avg": "red",
        }

        self.all_flights_duration = []
        self.all_flights_name = []
        self.all_x_values = []
        self.all_y_values = []
        self.all_z_values = []
        self.all_distance_travelled = []
        self.total_good_flights = 0
        self.total_flights_overall = 0

    def read_flight_data(self, json_path):
        """Read JSON flight data into a DataFrame."""
        with open(json_path, "r") as f:
            return pd.DataFrame(json.load(f))

    def plot_pie_chart(self, values, labels, colors, title, save_path):
        """Create and save a pie chart with counts and percentages."""
        total = sum(values)

        def fmt(pct, all_vals):
            count = int(round(pct * total / 100.0))
            return f"{count} ({pct:.1f}%)"

        plt.figure(figsize=(5, 5))
        plt.pie(
            values, labels=labels, autopct=lambda pct: fmt(pct, values), colors=colors
        )
        plt.title(title)
        plt.savefig(save_path)
        plt.close()

    def plot_flight_data(
        self,
        file_names,
        durations,
        distances,
        x_vals,
        y_vals,
        z_vals,
        avg_duration,
        title,
        save_path,
        shape=(12, 8),
        distance_plot=True,
    ):
        """Create bar + line plot for flight durations and X/Y/Z values."""
        plt.figure(figsize=shape)

        ax1 = plt.gca()
        ax1.bar(
            file_names,
            durations,
            color=self.colors["duration"],
            label="Flight Duration",
        )
        ax1.axhline(
            y=avg_duration,
            color=self.colors["avg"],
            linestyle="--",
            label="Average Duration",
        )
        ax1.set_xlabel("Flight File Name")
        ax1.set_ylabel("Flight Duration (minutes)")
        ax1.set_title(title)
        ax1.set_xticks(range(len(file_names)))
        ax1.set_xticklabels(file_names, rotation=45, ha="right")
        ax1.grid(axis="y")

        ax2 = ax1.twinx()
        if distance_plot:
            ax2.plot(
                file_names,
                distances,
                label="Distance",
                marker="o",
                color=self.colors["distance"],
            )
            ax2.set_ylabel("Distance Values")
            save_path = save_path + "_distance.png"
        else:
            ax2.plot(
                file_names, x_vals, label="X Values", marker="o", color=self.colors["x"]
            )
            ax2.plot(
                file_names, y_vals, label="Y Values", marker="o", color=self.colors["y"]
            )
            ax2.plot(
                file_names, z_vals, label="Z Values", marker="o", color=self.colors["z"]
            )
            ax2.set_ylabel("X / Y / Z Values")
            save_path = save_path + "_coordinates.png"

        # Merge legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # Main Processing
    # =========================
    def main(self):
        for dir_name in os.listdir(self.base_dir):
            print(f"Processing directory: {dir_name}")
            dir_path = os.path.join(self.base_dir, dir_name)

            if not os.path.isdir(dir_path):
                continue

            trajectory_dir = os.path.join(dir_path, "trajectory_report_data")
            if not os.path.exists(trajectory_dir):
                print(f"Report data directory does not exist: {trajectory_dir}")
                continue

            # Read flight_data.json
            json_path = os.path.join(trajectory_dir, "flight_data.json")
            df = self.read_flight_data(json_path)

            minutes, file_names = [], []
            x_vals, y_vals, z_vals = [], [], []
            distance_travelled = []

            # Directory containing good flight images
            image_dir = os.path.join(trajectory_dir, "images")
            good_traj_dir = os.path.join(image_dir, "good_trajectories")

            with open(
                os.path.join(good_traj_dir, "error_good_paths.txt"), "w"
            ) as error:
                for img_file in os.listdir(good_traj_dir):
                    if not img_file.endswith(".png"):
                        continue

                    # Match image with flight data entry
                    match_index = df.index[df["file_name"] == img_file[:-4]].tolist()
                    if not match_index:
                        print(f"No matching entry found for {img_file[:-4]}")
                        # ! Added this functionality (rejecting the file that are not in json data) in the csv filtering script. But still check once.
                        error.write(f"No matching entry found for {img_file}\n")
                        continue

                    csv_path = os.path.join(dir_path, img_file[:-4] + ".csv")
                    try:
                        csv_data = pd.read_csv(
                            csv_path, on_bad_lines="skip", low_memory=False
                        )
                    except Exception as e:
                        print(f"Failed to read {csv_path}: {e}")
                        continue

                    # Convert GPS to NED coordinates
                    df_ned = FlightDataHelper.gps_to_ned(csv_data)

                    # Absolute max values
                    abs_max_x = df_ned["x"].abs().max()
                    abs_max_y = df_ned["y"].abs().max()
                    abs_max_z = df_ned["z"].abs().max()

                    # Flight duration
                    idx = match_index[0]
                    duration = df["flight_duration_min"].iloc[idx]

                    if duration is None or duration <= 0:
                        print(
                            f"Invalid duration for {os.path.join(dir_name, img_file[:-4] + '.csv')}: {duration}"
                        )
                        continue

                    distance = df["distance_travelled"].iloc[idx]
                    if distance is None or distance <= 0:
                        print(
                            f"Invalid distance for {os.path.join(dir_name, img_file[:-4] + '.csv')}: {distance}"
                        )
                        continue

                    # Append per-directory data
                    minutes.append(duration)
                    file_names.append(img_file[:-4])
                    x_vals.append(abs_max_x)
                    y_vals.append(abs_max_y)
                    z_vals.append(abs_max_z)
                    distance_travelled.append(distance)

                    # Append to global data
                    self.all_flights_duration.append(duration)
                    self.all_flights_name.append(img_file[:-4])
                    self.all_x_values.append(abs_max_x)
                    self.all_y_values.append(abs_max_y)
                    self.all_z_values.append(abs_max_z)
                    self.all_distance_travelled.append(distance)

                self.total_good_flights += len(minutes)
                self.total_flights_overall += len(df)

                # Pie chart: Good vs Bad flights (per directory)
                bad_flights = len(df) - len(minutes)
                self.plot_pie_chart(
                    [len(minutes), bad_flights],
                    ["Good Flights", "Bad Flights"],
                    [self.colors["good"], self.colors["bad"]],
                    f"Good vs Bad Flights in {dir_name}",
                    os.path.join(image_dir, "flight_quality_pie.png"),
                )

                if minutes:
                    avg_duration = np.mean(minutes)
                    self.plot_flight_data(
                        file_names,
                        minutes,
                        distance_travelled,
                        x_vals,
                        y_vals,
                        z_vals,
                        avg_duration,
                        f"Flight Durations in {dir_name}",
                        os.path.join(image_dir, "flight_durations"),
                    )

                print(f"Total flights in {dir_name}: {len(df)}")
                print(f"Good flights in {dir_name}: {len(minutes)}")
                print("=" * 80)

        # Overall pie chart
        total_bad_flights = self.total_flights_overall - self.total_good_flights
        self.plot_pie_chart(
            [self.total_good_flights, total_bad_flights],
            ["Good Flights", "Bad Flights"],
            [self.colors["good"], self.colors["bad"]],
            "Good vs Bad Flights - Overall",
            os.path.join(self.base_dir, "flight_quality_overall_pie.png"),
        )

        print("Processing complete.")
