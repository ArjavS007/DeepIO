import pandas as pd
import pymap3d as pm
import matplotlib.pyplot as plt
import torch


class FlightDataHelper:
    def __init__(self):
        pass

    @staticmethod
    def return_flight_duration(df):
        start_time = str(df["GPS Time"].iloc[0]).zfill(6)
        end_time = str(df["GPS Time"].iloc[-1]).zfill(6)

        start_seconds = (
            int(start_time[0:2]) * 3600
            + int(start_time[2:4]) * 60
            + int(start_time[4:6])
        )
        end_seconds = (
            int(end_time[0:2]) * 3600 + int(end_time[2:4]) * 60 + int(end_time[4:6])
        )

        # Handle midnight wrap
        if end_seconds < start_seconds:
            end_seconds += 24 * 3600

        duration_seconds = end_seconds - start_seconds
        duration_minutes = duration_seconds // 60
        return duration_minutes, duration_seconds

    @staticmethod
    def gps_to_ned(df, check=False):
        valid_rows = df[
            (df["GPS Lat"] != 0) & (df["GPS Lon"] != 0) & (df["Altitude"] != 0)
        ]
        if valid_rows.empty:
            return pd.DataFrame(columns=["x", "y", "z"])

        df = df[valid_rows.index[0] :].reset_index()

        lat_ref, lon_ref, alt_ref = df[["GPS Lat", "GPS Lon", "Altitude"]].iloc[0]

        ned_coords = [
            pm.geodetic2ned(
                row["GPS Lat"],
                row["GPS Lon"],
                row["Altitude"],
                lat_ref,
                lon_ref,
                alt_ref,
            )
            for _, row in df.iterrows()
        ]

        ned_df = pd.DataFrame(
            [(n, e, -d) for n, e, d in ned_coords], columns=["x", "y", "z"]
        )

        if check:
            ned_df["group"] = df["group"]
            ned_df["Mode"] = df["Mode"]

        return ned_df

    @staticmethod
    def generate_trajectory_image(df, output_image_full_path):
        fig = plt.figure(figsize=(14, 6))

        # Extract start and end points
        start_point = (df["x"].iloc[0], df["y"].iloc[0], df["z"].iloc[0])
        end_point = (df["x"].iloc[-1], df["y"].iloc[-1], df["z"].iloc[-1])

        # 1️⃣ Line plot
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot(df["x"], df["y"], df["z"], color="black", lw=1.5, label="Trajectory")
        ax1.scatter(*start_point, color="green", s=50, label="Start", zorder=5)
        ax1.scatter(*end_point, color="red", s=50, label="End", zorder=5)
        ax1.set_xlabel("North (m)")
        ax1.set_ylabel("East (m)")
        ax1.set_zlabel("Down (m)")
        ax1.set_title("Trajectory (Line Plot)")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2️⃣ Scatter plot
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(
            df["x"], df["y"], df["z"], c="blue", s=10, label="Points", alpha=0.7
        )
        ax2.scatter(*start_point, color="green", s=80, label="Start", zorder=5)
        ax2.scatter(*end_point, color="red", s=80, label="End", zorder=5)
        ax2.set_xlabel("North (m)")
        ax2.set_ylabel("East (m)")
        ax2.set_zlabel("Down (m)")
        ax2.set_title("Trajectory (Scatter Plot)")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_image_full_path, dpi=300)
        plt.close(fig)

    @staticmethod
    def filter_csv(df):
        if df.empty:
            return False, False

        # Condition 1: All values within ±50 (for x, y, z)
        within_50 = (
            df[["x", "y", "z"]].apply(lambda col: col.between(-50, 50).all()).any()
        )

        # Condition 2: Any value outside ±5000 (for z)
        outside_3000 = (~df["z"].between(-3000, 3000)).any()

        # Condition 3: Any sudden change in the x, y, z values (more than the threshold)
        # ! removing this condition as we are training with segments now.
        # sudden_change = (
        #     (df["x"].diff().fillna(0).abs() > 50)
        #     | (df["y"].diff().fillna(0).abs() > 50)
        #     | (df["z"].diff().fillna(0).abs() > 50)
        # ).any()

        # return bool(within_50), bool(outside_5000), bool(sudden_change)
        return bool(within_50), bool(outside_3000)

    @staticmethod
    def to_datetime(df):
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

    @staticmethod
    def add_delta(df):
        df["delta_x"] = df["x"].diff().fillna(0)
        df["delta_y"] = df["y"].diff().fillna(0)
        df["delta_z"] = df["z"].diff().fillna(0)

    @staticmethod
    def get_last_non_zero_or_last(group):
        non_zero_row = group[(group != 0).any(axis=1)]
        if not non_zero_row.empty:
            return non_zero_row.iloc[-1]
        else:
            return group.iloc[-1]

    # Max position error
    @staticmethod
    def get_max_pe(y, y_hat):
        # return np.max(np.linalg.norm(np.array(y)-np.array(y_hat),axis=1))
        return torch.max(torch.norm(y - y_hat, dim=1))

    @staticmethod
    def get_max_pe_norm(y, y_hat):
        """Calculates the normalized maximum prediction error by dividing the
        largest error by the cumulative distance up to that point.
        """
        max_pe_index = torch.argmax(torch.norm(y - y_hat, dim=1))
        dist = torch.sum(torch.norm(y[1 : max_pe_index + 1] - y[:max_pe_index], dim=1))
        max_pe = torch.max(torch.norm(y - y_hat, dim=1))
        return max_pe / dist

    @staticmethod
    def get_mean_pe(y, y_hat):
        # return np.mean(np.linalg.norm(np.array(y)-np.array(y_hat),axis=1))
        return torch.mean(torch.norm(y - y_hat, dim=1))

    @staticmethod
    def get_mean_pe_norm(y, y_hat):
        dist = torch.sum(torch.norm(y[1:] - y[:-1], dim=1))
        mean_pe = torch.mean(torch.norm(y - y_hat, dim=1))
        return mean_pe / dist
