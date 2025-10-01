from models import LSTMModelV3
from trainer import SimpleModelTrainerCpu
import os
from utils import Utils
import torch
# import subprocess

Utils.print_gpu_info()
# subprocess.run("nvidia-smi")

csv_files_dir = "/home/arjav.singh/DeepIO/Production_Data/SIDM/csvs"
val_txt_file = (
    "/home/arjav.singh/DeepIO/Production_Data/SIDM/src_148_code/selected_val_traj.txt"
)
with open(val_txt_file, "r") as f:
    val_png_files = f.read().splitlines()

val_flights = []
for paths in val_png_files:
    paths = paths.split("/")
    dirname = paths[7]
    filename = paths[-1].replace(".png", ".csv")
    full_path = os.path.join(csv_files_dir, dirname, filename)
    val_flights.append(full_path)

print(f"Number of val flights: {len(val_flights)}")

model = LSTMModelV3(in_dim=11, hidden_size=1400, output_size=2)  #  dropout_prob=0.3
checkpoint_path = "/home/arjav.singh/DeepIO/Production_Data/SIDM/after_training/GPU Train 3-20250910-141810/checkpoints/best_model.pt"
checkpoint = torch.load(
    checkpoint_path,
    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
model.load_state_dict(checkpoint["model_state_dict"])

trainer = SimpleModelTrainerCpu(
    model,
    val_flights,
    val_flights,
    val_flights,
    do_windowing=True,
    window_size=200,
    epochs=300,
    patience=45,
    run_name="debug_o12_const_low_lr",
    lr=0.00005,
    pad_training_data=False,
    pad_testing_data=False,
    clip_gradients=True,
    description="Dropout",  # Warmup + Cosine Annealing (0.0005 to 0.00005). BS = 1024'
    wandb_logging=False,
    batch_size=512,
    log_debug_data=True,
    checkpoint_path="/deep_io/runs/o15_fo13_const_low_lr-20240906-140232/checkpoints/0.0000237582.pt",
)

# trainer.train_validate()
trainer.validate()

# ec2.Instance('i-02a273abf0f8ff3d6').stop()

# ec2.terminate_instances(InstanceIds=['i-02a273abf0f8ff3d6'])


#
