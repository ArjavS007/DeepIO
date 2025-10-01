from models import LSTMModelV3
from trainer import SimpleModelTrainerCpu
import os
import torch
# import subprocess
# subprocess.run("nvidia-smi")

csv_files_dir = "/home/arjav.singh/DeepIO/Production_Data/SIDM/csvs"
test_txt_file = (
    "/home/arjav.singh/DeepIO/Production_Data/SIDM/src_148_code/selected_test_traj.txt"
)
with open(test_txt_file, "r") as f:
    test_png_files = f.read().splitlines()

test_flights = []
for paths in test_png_files:
    paths = paths.split("/")
    dirname = paths[7]
    filename = paths[-1].replace(".png", ".csv")
    full_path = os.path.join(csv_files_dir, dirname, filename)
    test_flights.append(full_path)

print(f"Number of test flights: {len(test_flights)}")

model = LSTMModelV3(in_dim=11, hidden_size=1400, output_size=2)  #  dropout_prob=0.3
checkpoint_path = "/home/arjav.singh/DeepIO/Production_Data/SIDM/after_training/GPU Train 3-20250910-141810/checkpoints/best_model.pt"
checkpoint = torch.load(
    checkpoint_path,
    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
model.load_state_dict(checkpoint["model_state_dict"])

trainer = SimpleModelTrainerCpu(
    model,
    test_flights,
    test_flights,
    test_flights,
    do_windowing=True,
    window_size=200,
    epochs=30,
    patience=5,
    run_name="GPU Test 3",
    lr=1e-5,
    lr_patience=7,
    last_lr_change_patience=5,
    use_lr_scheduler=True,
    pad_train_data=False,
    pad_val_data=False,
    clip_gradients=True,
    description="Testing the checkpoint 3",  # Warmup + Cosine Annealing (0.0005 to 0.00005). BS = 1024'
    wandb_logging=False,
    batch_size=256,
    log_debug_data=True,
)

trainer.test(0)
