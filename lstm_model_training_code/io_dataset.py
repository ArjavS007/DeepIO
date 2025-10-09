from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
import pandas as pd
import torch
import pymap3d as pm


class IODatasetCpu(Dataset):
    def __init__(self, flight_paths, window_size=200, check=False):
        self.flight_csv_paths = flight_paths

        self.window = window_size

        self.x = None
        self.y = None

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

        self.flight_last_record_index = []
        self.flight_start_record_index = []
        self.flight_origins = []

        # self.load_data_preproc_every_11_rec()
        self._len = 0
        self.check = check
        self.load_data_preproc_every_1_sec(self.check)

        self.currentFlightIndex = 0
        self.indexOffset = 0

        # self._len = len(self.x) - len(self.flight_csv_paths) * (self.window - 1)

        self.create_index_map()

        assert not self.x.isnan().any(), "x contains NaN values"
        assert not self.y.isnan().any(), "y contains NaN values"
        print("Non NaN values in x and y.")
        print(f"x.device: {self.x.device}")
        print(f"y.device: {self.y.device}")
        print(f"x.shape: {self.x.shape}")
        print(f"y.shape: {self.y.shape}")
        print(f"self.new_flight_started: {len(self.new_flight_started)}")

    def __len__(self):
        return self._len

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

    def get_non_hold_intervals(self, df, start, end):
        """
        Return index pairs (start, end) for intervals where 'Mode' is NOT 'HOLD/30/F'.
        Handles edge cases like starting at index 0 or ending at last row.
        """
        check_value = "OFF/0/V"
        check_mapped_value = self.mode_mapping[check_value]

        intervals = []
        in_non_hold = False
        interval_start = None

        for i in range(start, end):
            if df["Mode"].iloc[i] != check_mapped_value:
                if not in_non_hold:  # entering a non-hold region
                    in_non_hold = True
                    interval_start = i
            else:
                if in_non_hold:  # exiting a non-hold region
                    intervals.append((interval_start, i - 1))
                    in_non_hold = False

        # Edge case: if the sequence ended inside a non-hold region
        if in_non_hold:
            intervals.append((interval_start, end - 1))

        return intervals

    def load_data_preproc_every_1_sec(self, check=False):
        print("Loading data...")
        self.flight_start_record_index = [0]

        x = []
        y = []

        for i in tqdm(range(len(self.flight_csv_paths)), desc="Flight#", mininterval=5):
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

            sum_columns = ["Gp", "Gq", "Gr", "Ax", "Ay", "Az"]
            avg_columns = ["Bx", "By", "Bz", "Altitude"]

            df = pd.read_csv(
                self.flight_csv_paths[i], on_bad_lines="skip", low_memory=False
            )
            df = df[relevant_columns]

            df_non_zero = df[
                (df["GPS Lat"] != 0) & (df["GPS Lon"] != 0) & (df["GPS AGL"] != 0)
            ]
            df = df[df_non_zero.index[0] :]  # remove initial all-zero rows
            df = df.reset_index()

            self.parse_gps_time(df)
            df["group"] = df["timestamp_seconds"] - df["timestamp_seconds"].iloc[0]

            df["Mode"] = df["Mode"].map(self.mode_mapping)

            df_ned = df.copy(deep=True)

            agg_map = {col: "sum" for col in sum_columns}
            agg_map.update({col: "mean" for col in avg_columns})

            df = df.groupby(["group", "Mode"], as_index=False, sort=False).agg(agg_map)
            assert not df.isnull().values.any(), "df contains NaN values"

            # Collect the indices where there is a lag
            diff_df = df["group"].diff()
            mask = diff_df > 1
            lag_indices = list(
                mask[mask].index
            )  # ! remember to incorporate the 0th element.
            lag_indices = [0] + lag_indices + [len(diff_df)]

            df_in = df[input_columns].copy(deep=True)
            df_in["Altitude"] = df_in["Altitude"].diff()
            df_in.loc[df_in.index[0], "Altitude"] = 0

            df_ned = self.gps_to_ned(df_ned)
            df_ned = df_ned.groupby(["group", "Mode"]).apply(
                self.get_last_non_zero_or_last, include_groups=False
            )
            self.add_delta_NED(df_ned)
            df_delta_ned = df_ned[["delta_x", "delta_y", "delta_z"]]

            if len(df_in) != len(df_delta_ned):
                pass

            # check if df_in and df_delta_ned[['delta_x', 'delta_y', 'delta_z']] contains NaN values
            if df_in.isna().any().any():
                print(
                    f"df_in contains NaN values for flight {self.flight_csv_paths[i]}"
                )
                continue

            if df_delta_ned[["delta_x", "delta_y", "delta_z"]].isna().any().any():
                print(
                    f"df_delta_ned contains NaN values for flight {self.flight_csv_paths[i]}"
                )
                continue

            # Checking for lags and creating sequences accordingly
            for j in range(1, len(lag_indices)):
                # ! Since during differencing we subtract prev from curr means the curr index is after lag.
                prev = lag_indices[j - 1]
                curr = lag_indices[j]

                non_hold_indices = [(prev, curr)]

                if check:
                    non_hold_indices = self.get_non_hold_intervals(df_in, prev, curr)

                for prev, curr in non_hold_indices:
                    val = curr - prev
                    if val > 3 * self.window:
                        self._len += val - (self.window - 1)
                        x.extend(df_in.iloc[prev:curr].values.tolist())
                        y.extend(
                            df_delta_ned[["delta_x", "delta_y", "delta_z"]]
                            .iloc[prev:curr]
                            .values.tolist()
                        )

                        # record current flight's last record index
                        self.flight_last_record_index.append(len(x) - 1)

                        # record current flight's start record index
                        if j != len(lag_indices) - 1:
                            self.flight_start_record_index.append(len(x))

                        # record flight NED origin
                        self.flight_origins.append(
                            df_ned.iloc[prev][["x", "y", "z"]].values.tolist()
                        )

        self.x = torch.tensor(x, dtype=torch.float32).to("cpu")
        self.y = torch.tensor(y, dtype=torch.float32).to("cpu")

    def create_index_map(self):
        self.index_map = {}
        new_flight_started = []
        currentFlightIndex = 0
        indexOffset = 0

        for i in range(self._len):
            if (
                i + indexOffset + self.window - 1
                > self.flight_last_record_index[currentFlightIndex]
            ):
                currentFlightIndex += 1
                indexOffset += self.window - 1
                # new_flight_started[i] = True # for second flight onwards
                new_flight_started.append(True)
            else:
                if not new_flight_started:  # for very record i.e. start of first flight
                    new_flight_started.append(True)
                else:  # for subsequent flights
                    new_flight_started.append(False)

            self.index_map[i] = i + indexOffset

        self.new_flight_started = torch.tensor(new_flight_started, dtype=torch.bool).to(
            "cpu"
        )

    def __getitem__(self, index):
        if index < 0 or index >= self._len:
            raise StopIteration("Index out of bounds")

        i = self.index_map[index]

        return (
            self.x[i : i + self.window],
            self.y[i + self.window - 1],
            self.new_flight_started[index],
        )

    def __old_getitem__(self, index):
        if index < 0 or index >= self._len:
            raise StopIteration("Index out of bounds")

        if (
            index + self.indexOffset + self.window - 1
            >= self.flight_last_record_index[self.currentFlightIndex]
        ):
            if self.currentFlightIndex < len(self.flight_last_record_index) - 1:
                self.currentFlightIndex += 1
                self.indexOffset += self.window
            else:
                self.currentFlightIndex = 0
                self.indexOffset = 0
        _x = self.x[index + self.indexOffset : index + self.indexOffset + self.window]
        _y = self.y[index + self.indexOffset + self.window - 1]
        return _x, _y
