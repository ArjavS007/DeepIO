from models import LSTMModelV3
from transformer_models import DecoderOnlyTransformer
from trainer import SimpleModelTrainerCpu
import os
from utils import Utils
import random
import subprocess


Utils.print_gpu_info()

subprocess.run("nvidia-smi")


# csv_files_dir = "/home/perception/Projects/DeepIO/Production_Data/new_SIDM/csvs"
# good_csvs = []
# for dir in os.listdir(csv_files_dir):
#     dir_path = os.path.join(csv_files_dir, dir)
#     if not os.path.isdir(dir_path):
#         continue

#     traj_dir = os.path.join(
#         dir_path, "trajectory_report_data", "images", "good_trajectories"
#     )
#     if not os.path.isdir(traj_dir):
#         continue

#     for f in os.listdir(traj_dir):
#         if f.endswith(".png"):
#             csv_path = os.path.join(dir_path, f.replace(".png", ".csv"))
#             good_csvs.append(csv_path)

# test_txt_file = "/home/perception/Projects/DeepIO/Production_Data/new_SIDM/src_code_training/test_data_traj/selected_test_traj.txt"
# with open(test_txt_file, "r") as f:
#     test_png_files = f.read().splitlines()

# test_flights = []
# for paths in test_png_files:
#     paths = paths.split("/")
#     dirname = paths[7]
#     filename = paths[-1].replace(".png", ".csv")
#     full_path = os.path.join(csv_files_dir, dirname, filename)
#     if full_path in good_csvs:
#         good_csvs.remove(full_path)
#     test_flights.append(full_path)

# print(f"Number of test flights: {len(test_flights)}")
# print(f"Number of remaining good flights: {len(good_csvs)}")


# val_txt_file = "/home/perception/Projects/DeepIO/Production_Data/new_SIDM/src_code_training/val_data_traj/selected_val_traj.txt"
# with open(val_txt_file, "r") as f:
#     val_png_files = f.read().splitlines()

# val_flights = []
# for paths in val_png_files:
#     paths = paths.split("/")
#     dirname = paths[7]
#     filename = paths[-1].replace(".png", ".csv")
#     full_path = os.path.join(csv_files_dir, dirname, filename)
#     if full_path in good_csvs:
#         good_csvs.remove(full_path)
#     val_flights.append(full_path)

# print(f"Number of val flights: {len(val_flights)}")
# print(f"Number of remaining good flights: {len(good_csvs)}")

# flight_paths = good_csvs[:]
# random.shuffle(flight_paths)

# n = len(flight_paths)
# n_train = int(0.8 * n)
# n_val = int(0.2 * n)

# train_flights = flight_paths[:]

# train_flights = flight_paths[:10]
# val_flights = flight_paths[11:20]
# test_flights = flight_paths[20:30]

csv_files_dir = "/media/perception/DATADRIVE2/DeepIO/switch_series/data/csvs"
dir_paths = []  # <--- Leave few directores for blind test
for dir in os.listdir(csv_files_dir):
    dir_path = os.path.join(csv_files_dir, dir)

    if not os.path.isdir(dir_path):
        continue

    dir_paths.append(dir_path)

random.shuffle(dir_paths)
train_dirs = dir_paths[:-1]
print(f"Count of UAVs for training: {len(train_dirs)}")

test_dirs = dir_paths[-1:]
print(f"The testing directory is {test_dirs}")
print(f"Count of UAVs for testing: {len(test_dirs)}")

good_train_csvs = []
for dir_path in train_dirs:
    traj_dir = os.path.join(
        dir_path, "trajectory_report_data", "images", "good_trajectories"
    )
    if not os.path.isdir(traj_dir):
        continue

    for f in os.listdir(traj_dir):
        if f.endswith(".png"):
            csv_path = os.path.join(dir_path, f.replace(".png", ".csv"))
            good_train_csvs.append(csv_path)

good_test_csvs = []
for dir_path in test_dirs:
    traj_dir = os.path.join(
        dir_path, "trajectory_report_data", "images", "good_trajectories"
    )
    if not os.path.isdir(traj_dir):
        continue

    for f in os.listdir(traj_dir):
        if f.endswith(".png"):
            csv_path = os.path.join(dir_path, f.replace(".png", ".csv"))
            good_test_csvs.append(csv_path)

frac_of_flights_for_training = 1
frac_of_flights_for_testing = 1
print(f"Number of training flights: {len(good_train_csvs)}")
print(f"Number of testing flights: {len(good_test_csvs)}")

# Separate out the train and val data
flights = good_train_csvs[:]
random.shuffle(flights)
train_flight_paths = flights[: int(frac_of_flights_for_training * len(good_train_csvs))]

# Separate out the fraction of test data
flights = good_test_csvs[:]
random.shuffle(flights)
test_flight_paths = flights[: int(frac_of_flights_for_testing * len(good_test_csvs))]


n = len(train_flight_paths)
n_train = int(0.7 * n)
n_val = int(0.15 * n)

train_flights = train_flight_paths[:n_train]
val_flights = train_flight_paths[n_train : n_train + n_val]
test_flights = test_flight_paths[:]


# model = LSTMModelV3(in_dim=11, hidden_size=1400, output_size=2)
model = DecoderOnlyTransformer(in_dim=11, model_dim=1000, out_dim=2, num_layers=3)

trainer = SimpleModelTrainerCpu(
    model,
    train_flights,
    val_flights,
    test_flights,
    do_windowing=True,
    window_size=100,
    epochs=1000,
    patience=10,
    run_name="GPU Train 1",
    lr=5e-4,
    lr_patience=7,
    last_lr_change_patience=5,
    use_lr_scheduler=True,
    pad_train_data=False,
    pad_val_data=False,
    clip_gradients=True,
    description=f"Testing directory is - {test_dirs}",
    wandb_logging=True,
    batch_size=256,
    log_debug_data=True,
    results_dir="/home/perception/Projects/DeepIO/Production_Data/new_SIDM/transformers_training_results",
)

trainer.train_test()
