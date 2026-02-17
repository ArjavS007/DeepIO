from models import LSTMModelV6
from transformer_models import DecoderOnlyTransformer
from trainer import SimpleModelTrainerCpu
import os
from utils import Utils
import random
import subprocess
import wandb
from dotenv import load_dotenv
load_dotenv()
Utils.print_gpu_info()
subprocess.run("nvidia-smi")

csv_files_dir = "/home/ubuntu/DeepIO/switch/data/csvs"
dir_paths = []

# -----------------------------
# COLLECT ALL DIRECTORY PATHS
# -----------------------------
for dir in os.listdir(csv_files_dir):
    dir_path = os.path.join(csv_files_dir, dir)
    if os.path.isdir(dir_path):
        dir_paths.append(dir_path)

train_dirs = dir_paths[:-2]
test_dirs = dir_paths[-2:]

print(f"Count of UAVs for training: {len(train_dirs)}")
print(f"Count of UAVs for testing: {len(test_dirs)}")


# -----------------------------
# SUBSAMPLING FUNCTION (15%)
# -----------------------------
def get_subsampled_csvs(directories, fraction=0.15):
    collected = []
    for dir_path in directories:
        traj_dir = os.path.join(
            dir_path, "trajectory_report_data", "images", "good_trajectories"
        )
        if not os.path.isdir(traj_dir):
            continue

        png_files = [f for f in os.listdir(traj_dir) if f.endswith(".png")]

        # convert each PNG → CSV path
        csv_paths = [
            os.path.join(dir_path, f.replace(".png", ".csv")) for f in png_files
        ]

        # sample 15% from each directory
        k = max(0, int(len(csv_paths) * fraction))
        sampled = random.sample(csv_paths, k=k)
        collected.extend(sampled)

    return collected


# -----------------------------
# SUBSAMPLE TRAIN & TEST
# -----------------------------
good_train_csvs = get_subsampled_csvs(train_dirs, fraction=1.0)
good_test_csvs = get_subsampled_csvs(test_dirs, fraction=1.0)

print(f"Training flights after subsampling: {len(good_train_csvs)}")
print(f"Testing flights after subsampling: {len(good_test_csvs)}")

# -----------------------------
# SHUFFLE AND TRAIN/VAL SPLIT
# -----------------------------
random.shuffle(good_train_csvs)
random.shuffle(good_test_csvs)

n = len(good_train_csvs)
n_train = int(0.9 * n)

train_flights = good_train_csvs[:n_train]
val_flights = good_train_csvs[n_train: ]
test_flights = good_test_csvs[:]  # keep all test

print("Final Split:")
print(f"Train: {len(train_flights)}")
print(f"Val: {len(val_flights)}")
print(f"Test: {len(test_flights)}")

# -----------------------------
# MODEL + TRAINER SETUP
# -----------------------------
load_dotenv()
wandb_api_key = os.getenv("wandb_api_key")
if not wandb_api_key:
    raise ValueError("WandB API key not found in environment variables.")
wandb.login(key=wandb_api_key)

model = LSTMModelV6(in_dim=11, hidden_size=1400, num_layers=1, output_size=2)

trainer = SimpleModelTrainerCpu(
    model,
    train_flights,
    val_flights,
    test_flights,
    do_windowing=True,
    window_size=200,
    epochs=1000,
    patience=10,
    run_name="Switch Train 20",
    lr=5e-4,
    lr_patience=7,
    last_lr_change_patience=5,
    use_lr_scheduler=True,
    pad_train_data=False,
    pad_val_data=False,
    clip_gradients=True,
    description="Training with 100% per-directory subsampling",
    wandb_logging=True,
    batch_size=256,
    log_debug_data=True,
    results_dir="/home/ubuntu/DeepIO/switch/results/training_results",
    tune=True,
    saved_train_data_path="train_data.pt",
    saved_val_data_path="val_data.pt",
    saved_test_data_path="test_data.pt",
)

trainer.train_test()

