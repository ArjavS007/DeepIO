from models import LSTMModelV3
from transformer_models import DecoderOnlyTransformer
from trainer import SimpleModelTrainerCpu
import os
from utils import Utils
import random
# import subprocess


Utils.print_gpu_info()

# subprocess.run("nvidia-smi")


csv_files_dir = "/home/arjav.singh/Projects/DeepIO/Production_Data/SIDM/csvs"
good_csvs = []
for dir in os.listdir(csv_files_dir):
    dir_path = os.path.join(csv_files_dir, dir)
    if not os.path.isdir(dir_path):
        continue

    traj_dir = os.path.join(
        dir_path, "trajectory_report_data", "images", "good_trajectories"
    )
    if not os.path.isdir(traj_dir):
        continue

    for f in os.listdir(traj_dir):
        if f.endswith(".png"):
            csv_path = os.path.join(dir_path, f.replace(".png", ".csv"))
            good_csvs.append(csv_path)

test_txt_file = "/home/arjav.singh/Projects/DeepIO/Production_Data/SIDM/src_code_training/test_data_traj/selected_test_traj.txt"
with open(test_txt_file, "r") as f:
    test_png_files = f.read().splitlines()

test_flights = []
for paths in test_png_files:
    paths = paths.split("/")
    dirname = paths[7]
    filename = paths[-1].replace(".png", ".csv")
    full_path = os.path.join(csv_files_dir, dirname, filename)
    if full_path in good_csvs:
        good_csvs.remove(full_path)
    test_flights.append(full_path)

print(f"Number of test flights: {len(test_flights)}")
print(f"Number of remaining good flights: {len(good_csvs)}")


val_txt_file = "/home/arjav.singh/Projects/DeepIO/Production_Data/SIDM/src_code_training/val_data_traj/selected_val_traj.txt"
with open(val_txt_file, "r") as f:
    val_png_files = f.read().splitlines()

val_flights = []
for paths in val_png_files:
    paths = paths.split("/")
    dirname = paths[7]
    filename = paths[-1].replace(".png", ".csv")
    full_path = os.path.join(csv_files_dir, dirname, filename)
    if full_path in good_csvs:
        good_csvs.remove(full_path)
    val_flights.append(full_path)

print(f"Number of val flights: {len(val_flights)}")
print(f"Number of remaining good flights: {len(good_csvs)}")

flight_paths = good_csvs[:]
random.shuffle(flight_paths)

n = len(flight_paths)
n_train = int(0.8 * n)
n_val = int(0.2 * n)

train_flights = flight_paths[:]

train_flights = flight_paths[:1]
val_flights = flight_paths[1:2]
test_flights = flight_paths[2:3]


# model = LSTMModelV3(in_dim=11, hidden_size=100, output_size=2)
model = DecoderOnlyTransformer(in_dim=11, model_dim=1000, out_dim=2, num_layers=3)

trainer = SimpleModelTrainerCpu(
    model,
    train_flights,
    val_flights,
    test_flights,
    do_windowing=True,
    window_size=100,
    epochs=2,
    patience=5,
    run_name="Transformer Trial 1",
    lr=5e-4,
    lr_patience=7,
    last_lr_change_patience=5,
    use_lr_scheduler=True,
    pad_train_data=False,
    pad_val_data=False,
    clip_gradients=True,
    description="GPU instance",
    wandb_logging=False,
    batch_size=256,
    log_debug_data=True,
    results_dir="/home/arjav.singh/Projects/DeepIO/Production_Data/SIDM/training_results",
)

trainer.train_test()
