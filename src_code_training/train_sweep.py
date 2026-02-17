import wandb
import argparse
import subprocess
from models import LSTMModelV6
from trainer import SimpleModelTrainerCpu
from utils import Utils
from dotenv import load_dotenv
import os

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--num_blocks", type=int)
    parser.add_argument("--lstm_layers", type=int)

    parser.add_argument("--norm_type", type=str)
    parser.add_argument("--norm_position", type=str)

    parser.add_argument("--residuals", type=lambda x: x.lower() == "true")
    parser.add_argument("--residual_scale", type=float)

    parser.add_argument("--dropout_lstm", type=float)
    parser.add_argument("--dropout_output", type=float)

    parser.add_argument("--init_type", type=str)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--lr_patience", type=int)
    parser.add_argument("--last_lr_change_patience", type=int)

    return parser.parse_args()


def main():
    _ = parse_args()  # needed only so argparse consumes CLI args
    
    load_dotenv()
    wandb_api_key = os.getenv("wandb_api_key")
    if not wandb_api_key:
        raise ValueError("WandB API key not found in environment variables.")
    wandb.login(key=wandb_api_key)
    wandb.init(project="DeepIO")
    cfg = wandb.config

    # -----------------------------
    # MODEL
    # -----------------------------
    model = LSTMModelV6(
        in_dim=11,
        hidden_size=cfg.hidden_size,
        output_size=2,
        num_blocks=cfg.num_blocks,
        lstm_layers=cfg.lstm_layers,
        norm_type=cfg.norm_type,
        norm_position=cfg.norm_position,
        residuals=cfg.residuals,
        residual_scale=cfg.residual_scale,
        dropout_lstm=cfg.dropout_lstm,
        dropout_output=cfg.dropout_output,
        init_type=cfg.init_type,
    )

    wandb.log({
        "num_params": sum(p.numel() for p in model.parameters())
    })

    # -----------------------------
    # TRAINER
    # -----------------------------
    trainer = SimpleModelTrainerCpu(
        model,
        None,
        None,
        None,
        do_windowing=True,
        window_size=cfg.window_size,
        epochs=1000,
        patience=cfg.patience,
        run_name=f"Sweep-LSTMModelV6-1",
        lr=cfg.lr,
        lr_patience=cfg.lr_patience,
        last_lr_change_patience=cfg.last_lr_change_patience,
        use_lr_scheduler=True,
        pad_train_data=False,
        pad_val_data=False,
        clip_gradients=True,
        description="W&B sweep run for LSTMModelV6",
        wandb_logging=True,
        batch_size=cfg.batch_size,
        log_debug_data=True,
        results_dir="/home/ubuntu/DeepIO/switch/results/sweep_results",
        tune=True,
        saved_train_data_path="train_data.pt",
        saved_val_data_path="val_data.pt",
        saved_test_data_path="test_data.pt",
    )

    trainer.train_test()


if __name__ == "__main__":
    Utils.print_gpu_info()
    subprocess.run("nvidia-smi")
    main()
