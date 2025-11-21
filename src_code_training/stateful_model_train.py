from models import LSTMModelV3, LSTMStateConverter
from trainer import SimpleModelTrainerCpu
import os
from utils import Utils
import torch
import random
# import subprocess


Utils.print_gpu_info()

# subprocess.run("nvidia-smi")

# --------------------------
# 🔹 DATA PREPARATION
# --------------------------

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

# train_flights = flight_paths[6:7]
# val_flights = flight_paths[4:5]
# test_flights = flight_paths[2:3]

# --------------------------
# 🔹 MODEL CONVERSION
# --------------------------

# 1. Load your trained stateless model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/home/arjav.singh/Projects/DeepIO/Production_Data/SIDM/training_results/GPU Train 14-20250929-124430/checkpoints/best_model.pt"
stateless_model = LSTMModelV3(in_dim=11, hidden_size=1400, output_size=2)
stateless_model.load_state_dict(
    torch.load(checkpoint_path, map_location=device)["model_state_dict"]
)

# 2. Convert it to stateful
converter = LSTMStateConverter()
stateful_model = converter.convert_to_stateful(
    stateless_model,
    device=device,
    stateful_mode=True,
)

# 3. Verify weights match
for (name1, p1), (name2, p2) in zip(
    stateless_model.named_parameters(), stateful_model.named_parameters()
):
    assert torch.allclose(p1, p2, atol=1e-6), (
        f"Weight mismatch in {name1}"
    )  # torch.allclose(p1, p2, atol=1e-6) can fail if the model was saved/loaded in mixed precision or slightly different floating formats (FP16 → FP32).If that happens, relax to atol=1e-5 or rtol=1e-4.
print("✅ All weights transferred successfully!")

# 4. Use the new model
stateful_model.reset_states(batch_size=256)

trainer = SimpleModelTrainerCpu(
    stateful_model,
    train_flights,
    val_flights,
    test_flights,
    do_windowing=True,
    window_size=100,
    epochs=30,
    patience=5,
    run_name="GPU Train 14 - Fine Tuning",
    lr=5e-4,
    lr_patience=7,
    last_lr_change_patience=5,
    use_lr_scheduler=True,
    pad_train_data=True,
    pad_val_data=True,
    pad_test_data=True,
    clip_gradients=True,
    description="Fine tuning the stateful model converted from stateless model trained for 30 epochs.",
    wandb_logging=True,
    batch_size=256,
    log_debug_data=True,
    results_dir="/home/arjav.singh/Projects/DeepIO/Production_Data/SIDM/training_results",
    is_stateful=True,
)

trainer.train_test()
