from torch.optim import AdamW
from datetime import datetime
import torch.nn as nn
from io_dataset import IODatasetCpu
import copy
import wandb
import os
import json
import torch
from utils import Utils, LossAdaptiveLR
from tqdm.autonotebook import tqdm
import sys
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.nn.utils.rnn import pad_sequence
# import subprocess

import torch.multiprocessing as mp
mp.set_start_method('spawn')


class SimpleModelTrainerCpu:
    def __init__(
        self,
        model,
        train_flight_log_path,
        val_flight_log_path,
        test_flight_log_path,
        use_lr_scheduler=False,
        log_debug_data=False,
        pad_train_data=False,
        do_windowing=True,
        shuffle=False,
        window_size=200,
        ret_NED=False,
        batch_size=1024,
        epochs=100,
        patience=45,
        lr_patience=10,
        last_lr_change_patience=7,
        lr=0.00005,
        lr_list=[0.00005, 0.000005, 0.0000005, 0.00000005],
        checkpoint_path=None,
        run_name=None,
        pad_val_data=True,
        pad_test_data=True,
        clip_gradients=False,
        wandb_logging=True,
        description=None,
        results_dir=None,
        is_stateful=False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = AdamW(params=self.model.parameters(), lr=lr)
        self.use_lr_scheduler = use_lr_scheduler

        if use_lr_scheduler:
            self.scheduler = LossAdaptiveLR(
                self.optimizer,
                lr_list=lr_list,
                patience=lr_patience,
                change_lr_patience=last_lr_change_patience,
            )
        # ! Specify the directory where all results (checkpoints, val outputs, debug plots etc.) will be saved
        self.cd_paren_dir_prefix = results_dir if results_dir else "./training_results"
        print("Results dir: ", self.cd_paren_dir_prefix)
        
        self.debug_data = None
        self.log_debug_data = log_debug_data
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.do_windowing = do_windowing
        self.pad_train_data = pad_train_data
        self.pad_val_data = pad_val_data
        self.pad_test_data = pad_test_data
        self.ret_NED = ret_NED
        self.clip_gradients = clip_gradients
        self.window_size = window_size
        self.loss_fn = nn.L1Loss()  # Paper uses WeightedMAE() since it predicts both delta position and velocities. L1Loss is (un-weighted) MAE. But we can simply stick with MSE.
        # self.loss_fn = nn.MSELoss() # Since the values of Y and y_hat are position diffs and are tiny decimals, MSE turns out to be very tiny and in fact zero in very first epoch
        self.best_val_loss = float("inf")
        self.best_val_epoch = 0
        self.stop_early = False
        # self.model_summary_printed = False
        self.run_name = run_name
        self.run_name_ts = run_name if run_name else ""
        self.run_name_ts += "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.wandb_logging = wandb_logging
        self.gpu_resource_info_logged = False
        self.train_data_y = None
        self.train_data_y_hat = None
        self.val_data_y = None
        self.val_data_y_hat = None
        self.is_stateful = is_stateful

        print(f"{'=' * 20} {self.run_name_ts} {'=' * 20}")

        if checkpoint_path is not None:
            epoch = self.load_checkpoint(
                checkpoint_path
            )  # Example filename with validation loss
            if epoch is not None:
                print("Loaded checkpoint from epoch {}".format(epoch))

        self.best_model = copy.deepcopy(self.model)
        self.stats = []

        # os.makedirs(f"../runs/{self.run_name_ts}", exist_ok=True)
        os.makedirs(
            f"{self.cd_paren_dir_prefix}/{self.run_name_ts}/checkpoints", exist_ok=True
        )
        os.makedirs(
            f"{self.cd_paren_dir_prefix}/{self.run_name_ts}/debug/train", exist_ok=True
        )
        os.makedirs(
            f"{self.cd_paren_dir_prefix}/{self.run_name_ts}/debug/val", exist_ok=True
        )
        os.makedirs(
            f"{self.cd_paren_dir_prefix}/{self.run_name_ts}/debug/test", exist_ok=True
        )

        self.printInfo()

        print(f"\nPreparing train dataset\n{'-' * 45}")
        self.train_data = IODatasetCpu(train_flight_log_path, window_size=window_size, check=True)
        print(
            f"Train dataset record (number of sequences) count: {len(self.train_data)}"
        )
        # ! Multiprocessing is causing error so removing it as of now. Using drop_last to remove the last batch. 
        # ! In stateful LSTM if the number of seq. in a batch is not equal to batch size then it cause error. 
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=False,
            # generator=torch.Generator(device=self.device),
            # num_workers=4,
            # prefetch_factor=16,
            # collate_fn=self.custom_collate_fn,
            # drop_last=True
        )

        print(f"\nPreparing val dataset\n{'-' * 45}")
        if val_flight_log_path:
            self.val_data = IODatasetCpu(
                val_flight_log_path, window_size=window_size, check=True
            )
            print(f"Val dataset record count: {len(self.val_data)}")
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_data,
                batch_size=self.batch_size,
                shuffle=False,
                # generator=torch.Generator(device=self.device),
                # num_workers=4,
                # prefetch_factor=16,
                # collate_fn=self.custom_collate_fn,
                # drop_last=True
            )

        print(f"\nPreparing testdataset\n{'-' * 45}")
        self.test_data = None
        if test_flight_log_path:
            self.test_data = IODatasetCpu(
                test_flight_log_path, window_size=window_size, check=True
            )
            print(f"Test dataset record count: {len(self.test_data)}")
            self.test_dataloader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                shuffle=False,
                # generator=torch.Generator(device=self.device),
                # num_workers=4,
                # prefetch_factor=16,
                # collate_fn=self.custom_collate_fn,
                # drop_last=True
            )

        if wandb_logging:
            load_dotenv()
            wandb_api_key = os.getenv("wandb_api_key")
            if not wandb_api_key:
                raise ValueError("WandB API key not found in environment variables.")
            wandb.login(key=wandb_api_key)
            wandb.init(
                project="DeepIO",
                name=self.run_name_ts,
                config={
                    "description": description,
                    "model": model,
                    "mode_size": sys.getsizeof(model),
                    "learning_rate": lr,
                    "epochs": epochs,
                    "window_size": window_size,
                    "batch_size": batch_size,
                    "patience": patience,
                    "pad_train_data": pad_train_data,
                    "pad_val_data": pad_val_data,
                    "train_flight_log_path": "split_1_train_flights",
                    "val_flight_log_path": "split_1_val_flights",
                    "checkpoint_path": checkpoint_path,
                    "model_name": str(model),
                },
            )
            wandb.watch(self.model, log="all", log_freq=10)

    def custom_collate_fn(self, batch):
        sequences, targets, new_flight_started = zip(*batch)
        if self.pad_train_data or self.pad_val_data or self.pad_test_data:
            sequences = pad_sequence(sequences, batch_first=True, padding_value=-1.0)
            targets = pad_sequence(targets, batch_first=True, padding_value=-1.0)
        else:
            sequences = torch.stack(sequences)
            targets = torch.stack(targets)
        new_flight_started = torch.tensor(new_flight_started, dtype=torch.bool)
        return sequences, targets, new_flight_started

    def train_test(self):
        self.train()
        self.test()

    def load_checkpoint(self, filename):
        """
        Loads the model state dictionary, optimizer state dictionary, and epoch from a checkpoint file.

        Args:
            model: The model to load the checkpoint into.
            optimizer: The optimizer to load the checkpoint into (optional).
            filename: The filename of the checkpoint file.

        Returns:
            int: The epoch number from the checkpoint (if saved).
        """
        assert os.path.exists(filename), "No checkpoint found at {}".format(filename)
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]

    def printInfo(self):
        print("Run name: ", self.run_name)
        print("Patience: ", self.patience)
        print("LR: ", self.lr)
        print("Window size: ", self.window_size)
        print("Batch size: ", self.batch_size)
        print("Model size: ", sys.getsizeof(self.model))
        print("Loss function: ", self.loss_fn)
        print("Pad val data: ", self.pad_val_data)
        print("Pad train data: ", self.pad_train_data)
        print("Optimizer: \n", self.optimizer)
        print("Model: \n", self.model)
        print(f"{'=' * 50}")

    def save_checkpoint(self, epoch, filename):
        """
        Saves the model state dictionary, optimizer state dictionary, and current epoch.

        Args:
            model: The model to save.
            optimizer: The optimizer used for training.
            epoch: The current training epoch.
            filename: The filename to save the checkpoint to (default: "checkpoint.pt").
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
        }
        print("checkpoint path: ", filename)
        torch.save(checkpoint, filename)

    def save_stats(self, stats, file_name):
        with open(
            os.path.join(f"{self.cd_paren_dir_prefix}", self.run_name_ts, file_name),
            "w",
        ) as f:
            f.write(json.dumps(stats) + "\n")

    def log_combined_plot(
        self,
        y,
        y_hat,
        distance,
        drift_pcts,
        flight_id,
        avg_drift,
        max_drift,
        avg_drift_pct,
        max_drift_pct,
        flight_pe,
        filename="combined.png",
    ):
        fig = plt.figure(figsize=(16, 6))

        # Create a GridSpec: 2 rows, 2 columns
        gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1])

        # --- Left subplot: 3D trajectory (span both rows) ---
        ax1 = fig.add_subplot(gs[:, 0], projection="3d")

        # True trajectory
        xt, yt, zt = (
            y[:, 0].detach().cpu(),
            y[:, 1].detach().cpu(),
            y[:, 2].detach().cpu(),
        )
        ax1.plot(xt, yt, zt, color="black", lw=2, label="True Trajectory")
        ax1.scatter(xt[0], yt[0], zt[0], color="green", s=60, label="True Start")
        ax1.scatter(xt[-1], yt[-1], zt[-1], color="red", s=60, label="True End")

        # Predicted trajectory
        xp, yp, zp = (
            y_hat[:, 0].detach().cpu(),
            y_hat[:, 1].detach().cpu(),
            y_hat[:, 2].detach().cpu(),
        )
        ax1.plot(
            xp, yp, zp, color="blue", lw=2, linestyle="--", label="Predicted Trajectory"
        )
        ax1.scatter(
            xp[0], yp[0], zp[0], color="cyan", s=80, label="Pred Start", marker="^"
        )
        ax1.scatter(
            xp[-1], yp[-1], zp[-1], color="orange", s=80, label="Pred End", marker="v"
        )

        ax1.set_xlabel("North (m)")
        ax1.set_ylabel("East (m)")
        ax1.set_zlabel("Down (m)")
        ax1.set_title("True vs Predicted Trajectories")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        # --- Right top subplot: Drift % ---
        ax2 = fig.add_subplot(gs[0, 1])
        if isinstance(drift_pcts, torch.Tensor):
            drift_pcts = drift_pcts.detach().cpu().numpy()
        ax2.plot(drift_pcts, label=f"Flight {flight_id}", color="blue")
        ax2.set_xlabel("Distance")
        ax2.set_ylabel("Drift %")
        ax2.set_title(f"Drift % per distance - Flight {flight_id}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- Right bottom subplot: Flight position error ---
        ax3 = fig.add_subplot(gs[1, 1])
        if isinstance(flight_pe, torch.Tensor):
            flight_pe = flight_pe.detach().cpu().numpy()
        ax3.plot(flight_pe, label=f"Flight {flight_id}", color="orange")
        ax3.set_xlabel("Index")
        ax3.set_ylabel("Position Error")
        ax3.set_title(f"Flight PE - Flight {flight_id}")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # --- Global annotations ---
        fig.text(0.05, 0.95, f"Number of Data Points: {len(y)}", fontsize=8)
        fig.text(0.05, 0.91, f"Average Drift: {avg_drift:.4f}", fontsize=8)
        fig.text(0.05, 0.87, f"Max Drift: {max_drift:.4f}", fontsize=8)
        fig.text(0.05, 0.83, f"Average Drift %: {avg_drift_pct:.4f}%", fontsize=8)
        fig.text(0.05, 0.79, f"Max Drift %: {max_drift_pct:.4f}%", fontsize=8)
        fig.text(0.05, 0.75, f"Distance travelled: {distance:.4f} m", fontsize=8)

        plt.tight_layout()

        # Save locally
        out_dir = f"{self.cd_paren_dir_prefix}/{self.run_name_ts}/debug"
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(f"{out_dir}/{filename}", dpi=300)
        plt.close(fig)

    # def train(self):
    #     print(f"\n{'=' * 20} Training {'=' * 20}")
    #     print("Run name: ", self.run_name_ts)
    #     # print(f"Dataloader device: {device}")
    #     total_samples = 0
    #     total_samples = 0

    #     Utils.print_current_time("Training start time: ")

    #     for epoch in tqdm(range(self.epochs), desc=">>>> epoch", mininterval=5):
    #         print("\n============================================", end="")

    #         if self.is_stateful:
    #             self.model.reset_states(self.batch_size)

    #         self.model.train()
    #         total_loss = 0
    #         total_samples = 0

    #         self.train_data_y = None
    #         self.train_data_y_hat = None
    #         self.new_flight_started = []

    #         for train_data in tqdm(
    #             self.train_dataloader, desc=">> train", mininterval=5
    #         ):
    #             X = train_data[0].to(self.device)  # .to(torch.float32).to(device)
    #             Y = train_data[1].to(self.device)  # .to(device)

    #             if not self.gpu_resource_info_logged:
    #                 print(
    #                     "---------------------------------- GPU resource info ----------------------------------"
    #                 )
    #                 print(f"size of X = {sys.getsizeof(X)}")
    #                 print(f"size of Y = {sys.getsizeof(Y)}")
    #                 print(f"size of model = {sys.getsizeof(self.model)}")
    #                 # subprocess.run("nvidia-smi")
    #                 print(
    #                     "---------------------------------------------------------------------------------------"
    #                 )
    #                 self.gpu_resource_info_logged = True

    #             if X.shape[0] != self.batch_size:
    #                 continue  # TODO to avoid RuntimeError: shape '[16, 1, 256]' is invalid for input of size 3328

    #             total_samples += self.batch_size  # 1 # TODO seems incorrect

    #             # st_proc = time.time()
    #             y_hat = self.model(X)
    #             # print("y_hat.shape = ", y_hat.shape)

    #             if self.log_debug_data:
    #                 if self.train_data_y is None:
    #                     self.train_data_y = Y
    #                     self.train_data_y_hat = y_hat
    #                     self.new_flight_started = train_data[2].tolist()
    #                 else:
    #                     self.train_data_y = torch.cat((self.train_data_y, Y), dim=0)
    #                     self.train_data_y_hat = torch.cat(
    #                         (self.train_data_y_hat, y_hat), dim=0
    #                     )
    #                     self.new_flight_started.extend(train_data[2].tolist())

    #             loss = self.loss_fn(y_hat, Y[:, :2])
    #             self.optimizer.zero_grad()  # TODO check if correct
    #             loss.backward()
    #             if self.clip_gradients:
    #                 torch.nn.utils.clip_grad_norm_(
    #                     self.model.parameters(), max_norm=1.0
    #                 )  # ! Gradient clipping - Why?
    #             self.optimizer.step()

    #             total_loss += loss.item()

    #         # generate y and y_hat trajectories for each training flight after each epoch to check if it is indeed overfitting
    #         flight_last_rec_idx = self.train_data.flight_last_record_index
    #         max_pe = 0  # max position error across all flights
    #         total_mean_pe = 0  # mean position error across all flights
    #         total_max_pe_1m = 0
    #         total_mean_pe_1m = 0
    #         flight_counter = 0

    #         # get indices where new flight started
    #         flight_start_rec_indices = [
    #             i for i, x in enumerate(self.new_flight_started) if x
    #         ]
    #         # for each flight, get its y and y_hat trajectory from self.train_data_y and self.train_data_y_hat based on flight_start_rec_idx
    #         for i in range(len(flight_start_rec_indices)):
    #             start_index = flight_start_rec_indices[i]
    #             end_index = (
    #                 flight_start_rec_indices[i + 1]
    #                 if i < len(flight_start_rec_indices) - 1
    #                 else len(self.train_data_y)
    #             )
    #             delta_y = self.train_data_y[start_index:end_index]
    #             delta_y_hat = self.train_data_y_hat[start_index:end_index]
    #             y_cumsum = torch.cumsum(delta_y, dim=0)
    #             y_hat_cumsum = torch.cumsum(delta_y_hat, dim=0)
    #             y_hat_cumsum = torch.cat(
    #                 (y_hat_cumsum, y_cumsum[:, 2:3]), dim=1
    #             )  # add z axis to y_hat_cumsum
    #             max_pe = max(max_pe, Utils.get_max_pe(y_cumsum, y_hat_cumsum))
    #             total_max_pe_1m += Utils.get_max_pe_norm(y_cumsum, y_hat_cumsum)
    #             total_mean_pe += Utils.get_mean_pe(y_cumsum, y_hat_cumsum)
    #             total_mean_pe_1m += Utils.get_mean_pe_norm(y_cumsum, y_hat_cumsum)

    #             avg_drift, max_drift, avg_drift_pct, max_drift_pct, drift_pcts = (
    #                 Utils.get_drift_percentage(y_cumsum, y_hat_cumsum)
    #             )

    #             if self.log_debug_data and epoch % 3 == 0:
    #                 self.log_combined_plot(
    #                     y_cumsum,
    #                     y_hat_cumsum,
    #                     drift_pcts,
    #                     flight_counter,
    #                     avg_drift,
    #                     max_drift,
    #                     avg_drift_pct,
    #                     max_drift_pct,
    #                     "train/{}_combined.png".format(flight_counter),
    #                 )
    #             flight_counter += 1

    #         avg_train_loss = total_loss / total_samples if total_samples else 0

    #         if self.val_data is not None:
    #             (
    #                 val_loss,
    #                 val_max_pe,
    #                 val_max_pe_1m,
    #                 val_mean_pe,
    #                 val_mean_pe_1m,
    #                 val_avg_drift_overall,
    #                 val_max_drift_overall,
    #             ) = self.val(epoch)
    #             if self.wandb_logging:
    #                 wandb.log(
    #                     {
    #                         "val_loss": val_loss,
    #                         "val_max_pe": val_max_pe,
    #                         "val_max_pe_1m": val_max_pe_1m,
    #                         "val_mean_pe": val_mean_pe,
    #                         "val_mean_pe_1m": val_mean_pe_1m,
    #                         "val_avg_drift_overall": val_avg_drift_overall,
    #                         "val_max_drift_overall": val_max_drift_overall,
    #                     }
    #                 )

    #             else:
    #                 (
    #                     val_loss,
    #                     val_max_pe,
    #                     val_max_pe_1m,
    #                     val_mean_pe,
    #                     val_mean_pe_1m,
    #                     val_avg_drift_overall,
    #                     val_max_drift_overall,
    #                 ) = 0, 0, 0, 0, 0, 0, 0

    #             # Same structure for test_data
    #             if self.test_data is not None:
    #                 (
    #                     test_loss,
    #                     test_max_pe,
    #                     test_max_pe_1m,
    #                     test_mean_pe,
    #                     test_mean_pe_1m,
    #                     test_avg_drift_overall,
    #                     test_max_drift_overall,
    #                 ) = self.test(epoch)

    #                 if self.wandb_logging:
    #                     wandb.log(
    #                         {
    #                             "test_loss": test_loss,
    #                             "test_max_pe": test_max_pe,
    #                             "test_max_pe_1m": test_max_pe_1m,
    #                             "test_mean_pe": test_mean_pe,
    #                             "test_mean_pe_1m": test_mean_pe_1m,
    #                             "test_avg_drift_overall": test_avg_drift_overall,
    #                             "test_max_drift_overall": test_max_drift_overall,
    #                         }
    #                     )

    #             else:
    #                 (
    #                     test_loss,
    #                     test_max_pe,
    #                     test_max_pe_1m,
    #                     test_mean_pe,
    #                     test_mean_pe_1m,
    #                     test_avg_drift_overall,
    #                     test_max_drift_overall,
    #                 ) = 0, 0, 0, 0, 0, 0, 0

    #         if self.use_lr_scheduler:
    #             self.scheduler.step(val_loss)
    #             current_lr = self.scheduler.get_last_lr()[0]
    #             print(
    #                 f"### Epoch {epoch + 1}/{self.epochs}, Learning Rate: {current_lr}"
    #             )

    #         lrs = {
    #             f"lr_group_{i}": param_group["lr"]
    #             for i, param_group in enumerate(self.optimizer.param_groups)
    #         }

    #         if self.wandb_logging:
    #             wandb.log(
    #                 {
    #                     "train_loss": avg_train_loss,
    #                     "train_max_pe": max_pe,
    #                     "train_max_pe_1m": (total_max_pe_1m / len(flight_last_rec_idx))
    #                     if flight_last_rec_idx
    #                     else 0,
    #                     "train_mean_pe": (total_mean_pe / len(flight_last_rec_idx))
    #                     if flight_last_rec_idx
    #                     else 0,
    #                     "train_mean_pe_1m": (
    #                         total_mean_pe_1m / len(flight_last_rec_idx)
    #                     )
    #                     if flight_last_rec_idx
    #                     else 0,
    #                     "train_avg_drift_pct": avg_drift_pct,
    #                     "train_max_drift_pct": max_drift_pct,
    #                     "epoch": epoch,
    #                     **lrs,
    #                 }
    #             )

    #         self.stats.append(
    #             {
    #                 "epoch": str(epoch),
    #                 "avg_train_loss": avg_train_loss,
    #                 "val_loss": val_loss,  # ,
    #                 # "mpe": mpe.item(),
    #                 # "tn_mpe": tn_mpe,
    #                 # "mve": mve
    #             }
    #         )

    #         self.save_stats(self.stats, "train_val_stats.json")

    #         print(
    #             f"*** Epoch {epoch} - Train loss:{avg_train_loss:.5f}, Val loss:{val_loss:.5f}, Test loss:{test_loss:.5f}, Val mean pe 1m: {val_mean_pe_1m:.5f}, Val max pe 1m: {val_max_pe_1m:.5f}, Test mean pe 1m: {test_mean_pe_1m:.5f}, Test max pe 1m: {test_max_pe_1m:.5f}"
    #         )  # , MPE:{mpe}, TN_MPE:{tn_mpe}

    #         if self.stop_early:
    #             print(
    #                 f"Early stopping at epoch {epoch} due to lack of improvement in validation loss"
    #             )
    #             break

    #         Utils.print_current_time("Epoch end time: ")

    #         if self.is_stateful:
    #             self.model.detach_states()  # important for stateful training

    def train(self):
        print(f"\n{'=' * 20} Training {'=' * 20}")
        print("Run name: ", self.run_name_ts)

        Utils.print_current_time("Training start time: ")

        for epoch in tqdm(range(self.epochs), desc=">>>> epoch", mininterval=5):
            print("\n============================================", end="")

            if self.is_stateful:
                self.model.reset_states(self.batch_size)

            self.model.train()
            total_loss = 0
            total_samples = 0

            self.train_data_y = None
            self.train_data_y_hat = None
            self.new_flight_started = []

            for train_data in tqdm(
                self.train_dataloader, desc=">> train", mininterval=5
            ):
                X = train_data[0].to(self.device)
                Y = train_data[1].to(self.device)

                if not self.gpu_resource_info_logged:
                    print(
                        "---------------------------------- GPU resource info ----------------------------------"
                    )
                    print(f"size of X = {sys.getsizeof(X)}")
                    print(f"size of Y = {sys.getsizeof(Y)}")
                    print(f"size of model = {sys.getsizeof(self.model)}")
                    print(
                        "---------------------------------------------------------------------------------------"
                    )
                    self.gpu_resource_info_logged = True

                if X.shape[0] != self.batch_size:
                    continue  # avoid RuntimeError: shape mismatch

                total_samples += self.batch_size

                # <<< FIX: Detach hidden states before forward pass to prevent graph accumulation
                if self.is_stateful:
                    self.model.detach_states()

                y_hat = self.model(X)

                # <<< FIX: Detach tensors when saving for debug/logging to prevent graph retention
                if self.log_debug_data:
                    Y_det = Y.detach().cpu()
                    y_hat_det = y_hat.detach().cpu()

                    if self.train_data_y is None:
                        self.train_data_y = Y_det
                        self.train_data_y_hat = y_hat_det
                        self.new_flight_started = train_data[2].tolist()
                    else:
                        self.train_data_y = torch.cat((self.train_data_y, Y_det), dim=0)
                        self.train_data_y_hat = torch.cat(
                            (self.train_data_y_hat, y_hat_det), dim=0
                        )
                        self.new_flight_started.extend(train_data[2].tolist())

                loss = self.loss_fn(y_hat, Y[:, :2])
                self.optimizer.zero_grad()
                loss.backward()

                if self.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                self.optimizer.step()

                total_loss += loss.item()

                # <<< FIX: Detach hidden states after optimizer step (safer for stateful LSTM)
                if self.is_stateful:
                    self.model.detach_states()

            # <<< FIX: Disable gradient tracking for metrics/logging
            with torch.no_grad():
                flight_last_rec_idx = self.train_data.flight_last_record_index
                max_pe = 0
                total_mean_pe = 0
                total_max_pe_1m = 0
                total_mean_pe_1m = 0
                flight_counter = 0

                flight_start_rec_indices = [
                    i for i, x in enumerate(self.new_flight_started) if x
                ]

                for i in range(len(flight_start_rec_indices)):
                    start_index = flight_start_rec_indices[i]
                    end_index = (
                        flight_start_rec_indices[i + 1]
                        if i < len(flight_start_rec_indices) - 1
                        else len(self.train_data_y)
                    )

                    delta_y = self.train_data_y[start_index:end_index]
                    delta_y_hat = self.train_data_y_hat[start_index:end_index]

                    y_cumsum = torch.cumsum(delta_y, dim=0)
                    y_hat_cumsum = torch.cumsum(delta_y_hat, dim=0)
                    y_hat_cumsum = torch.cat((y_hat_cumsum, y_cumsum[:, 2:3]), dim=1)
                    distance = torch.sum(torch.linalg.norm(delta_y, dim=1))
                    max_pe = max(max_pe, Utils.get_max_pe(y_cumsum, y_hat_cumsum))
                    total_max_pe_1m += Utils.get_max_pe_norm(y_cumsum, y_hat_cumsum)
                    total_mean_pe += Utils.get_mean_pe(y_cumsum, y_hat_cumsum)
                    total_mean_pe_1m += Utils.get_mean_pe_norm(y_cumsum, y_hat_cumsum)

                    avg_drift, max_drift, avg_drift_pct, max_drift_pct, drift_pcts = (
                        Utils.get_drift_percentage(y_cumsum, y_hat_cumsum)
                    )

                    if self.log_debug_data and epoch % 3 == 0:
                        self.log_combined_plot(
                            y_cumsum,
                            y_hat_cumsum,
                            distance,
                            drift_pcts,
                            flight_counter,
                            avg_drift,
                            max_drift,
                            avg_drift_pct,
                            max_drift_pct,
                            f"train/{flight_counter}_combined.png",
                        )

                    flight_counter += 1

            avg_train_loss = total_loss / total_samples if total_samples else 0

            # ============================ Validation ============================
            if self.val_data is not None:
                (
                    val_loss,
                    val_max_pe,
                    val_max_pe_1m,
                    val_mean_pe,
                    val_mean_pe_1m,
                    val_avg_drift_overall,
                    val_max_drift_overall,
                ) = self.val(epoch)
                if self.wandb_logging:
                    wandb.log(
                        {
                            "val_loss": val_loss,
                            "val_max_pe": val_max_pe,
                            "val_max_pe_1m": val_max_pe_1m,
                            "val_mean_pe": val_mean_pe,
                            "val_mean_pe_1m": val_mean_pe_1m,
                            "val_avg_drift_overall": val_avg_drift_overall,
                            "val_max_drift_overall": val_max_drift_overall,
                        }
                    )
            else:
                val_loss = val_max_pe = val_max_pe_1m = val_mean_pe = val_mean_pe_1m = (
                    val_avg_drift_overall
                ) = val_max_drift_overall = 0

            # ============================ Test ============================
            if self.test_data is not None:
                (
                    test_loss,
                    test_max_pe,
                    test_max_pe_1m,
                    test_mean_pe,
                    test_mean_pe_1m,
                    test_avg_drift_overall,
                    test_max_drift_overall,
                ) = self.test(epoch)

                if self.wandb_logging:
                    wandb.log(
                        {
                            "test_loss": test_loss,
                            "test_max_pe": test_max_pe,
                            "test_max_pe_1m": test_max_pe_1m,
                            "test_mean_pe": test_mean_pe,
                            "test_mean_pe_1m": test_mean_pe_1m,
                            "test_avg_drift_overall": test_avg_drift_overall,
                            "test_max_drift_overall": test_max_drift_overall,
                        }
                    )
            else:
                test_loss = test_max_pe = test_max_pe_1m = test_mean_pe = (
                    test_mean_pe_1m
                ) = test_avg_drift_overall = test_max_drift_overall = 0

            # ============================ Scheduler & Logging ============================
            if self.use_lr_scheduler:
                self.scheduler.step(val_loss)
                current_lr = self.scheduler.get_last_lr()[0]
                print(
                    f"### Epoch {epoch + 1}/{self.epochs}, Learning Rate: {current_lr}"
                )

            lrs = {
                f"lr_group_{i}": param_group["lr"]
                for i, param_group in enumerate(self.optimizer.param_groups)
            }

            if self.wandb_logging:
                wandb.log(
                    {
                        "train_loss": avg_train_loss,
                        "train_max_pe": max_pe,
                        "train_max_pe_1m": (total_max_pe_1m / len(flight_last_rec_idx))
                        if flight_last_rec_idx
                        else 0,
                        "train_mean_pe": (total_mean_pe / len(flight_last_rec_idx))
                        if flight_last_rec_idx
                        else 0,
                        "train_mean_pe_1m": (
                            total_mean_pe_1m / len(flight_last_rec_idx)
                        )
                        if flight_last_rec_idx
                        else 0,
                        "train_avg_drift_pct": avg_drift_pct,
                        "train_max_drift_pct": max_drift_pct,
                        "epoch": epoch,
                        **lrs,
                    }
                )

            self.stats.append(
                {
                    "epoch": str(epoch),
                    "avg_train_loss": avg_train_loss,
                    "val_loss": val_loss,
                }
            )

            self.save_stats(self.stats, "train_val_stats.json")

            print(
                f"*** Epoch {epoch} - Train loss:{avg_train_loss:.5f}, Val loss:{val_loss:.5f}, "
                f"Test loss:{test_loss:.5f}, Val mean pe 1m: {val_mean_pe_1m:.5f}, "
                f"Val max pe 1m: {val_max_pe_1m:.5f}, Test mean pe 1m: {test_mean_pe_1m:.5f}, "
                f"Test max pe 1m: {test_max_pe_1m:.5f}"
            )

            if self.stop_early:
                print(
                    f"Early stopping at epoch {epoch} due to lack of improvement in validation loss"
                )
                break

            Utils.print_current_time("Epoch end time: ")

            # <<< FIX: Detach final hidden states before next epoch
            if self.is_stateful:
                self.model.detach_states()

    def val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_samples = 0

            self.val_data_y = None
            self.val_data_y_hat = None
            self.new_flight_started = None

            for val_data in tqdm(self.val_dataloader, desc=">> val", mininterval=5):
                X = val_data[0].to(self.device)
                Y = val_data[1].to(self.device)

                if X.shape[0] != self.batch_size:
                    continue  # TODO to avoid RuntimeError: shape '[16, 1, 256]' is invalid for input of size 3328

                total_samples += self.batch_size  # TODO
                y_hat = self.model(X)

                loss = self.loss_fn(y_hat, Y[:, :2])
                total_loss += loss.item()

                if self.log_debug_data:
                    if self.val_data_y is None:
                        self.val_data_y = Y
                        self.val_data_y_hat = y_hat
                        self.new_flight_started = val_data[2].tolist()
                    else:
                        self.val_data_y = torch.cat((self.val_data_y, Y), dim=0)
                        self.val_data_y_hat = torch.cat(
                            (self.val_data_y_hat, y_hat), dim=0
                        )
                        self.new_flight_started.extend(val_data[2].tolist())

            # Early stopping (optional)
            val_loss = total_loss / total_samples

            # generate y and y_hat trajectories for each val flight after each epoch to check if it is indeed overfitting
            # flight_last_rec_idx = self.val_data.flight_last_record_index
            max_pe = 0  # max position error across all flights
            total_mean_pe = 0  # mean position error across all flights
            total_mean_pe_1m = 0
            total_max_pe_1m = 0
            flight_counter = 0
            per_flight_drift = []
            all_drift_pcts = []

            # get indices where new flight started
            flight_start_rec_indices = [
                i for i, x in enumerate(self.new_flight_started) if x
            ]

            for i in range(len(flight_start_rec_indices)):
                start_index = flight_start_rec_indices[i]
                end_index = (
                    flight_start_rec_indices[i + 1]
                    if i < len(flight_start_rec_indices) - 1
                    else len(self.val_data_y)
                )
                delta_y = self.val_data_y[start_index:end_index]
                delta_y_hat = self.val_data_y_hat[start_index:end_index]
                y_cumsum = torch.cumsum(delta_y, dim=0)
                y_hat_cumsum = torch.cumsum(delta_y_hat, dim=0)
                y_hat_cumsum = torch.cat((y_hat_cumsum, y_cumsum[:, 2:3]), dim=1)
                max_pe = max(max_pe, Utils.get_max_pe(y_cumsum, y_hat_cumsum))
                total_max_pe_1m = torch.max(
                    torch.tensor(total_max_pe_1m),
                    Utils.get_max_pe_norm(y_cumsum, y_hat_cumsum),
                )
                distance = torch.sum(torch.linalg.norm(delta_y, dim=1))
                total_mean_pe += Utils.get_mean_pe(y_cumsum, y_hat_cumsum)
                total_mean_pe_1m += Utils.get_mean_pe_norm(y_cumsum, y_hat_cumsum)
                flight_pe = Utils.get_pe(y_cumsum, y_hat_cumsum)
                avg_drift, max_drift, avg_drift_pct, max_drift_pct, drift_pcts = (
                    Utils.get_drift_percentage(y_cumsum, y_hat_cumsum)
                )
                per_flight_drift.append((avg_drift_pct, max_drift_pct))
                all_drift_pcts.append(drift_pcts)
                if self.log_debug_data and epoch % 3 == 0:
                    self.log_combined_plot(
                        y_cumsum,
                        y_hat_cumsum,
                        distance,
                        drift_pcts,
                        flight_counter,
                        avg_drift,
                        max_drift,
                        avg_drift_pct,
                        max_drift_pct,
                        flight_pe,
                        "val/{}_combined.png".format(flight_counter),
                    )
                flight_counter += 1

            avg_drift_overall = sum(d[0] for d in per_flight_drift) / len(
                per_flight_drift
            )
            max_drift_overall = sum(d[1] for d in per_flight_drift) / len(
                per_flight_drift
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_epoch = epoch
                self.best_model = copy.deepcopy(self.model)
                filepath = os.path.join(
                    f"{self.cd_paren_dir_prefix}",
                    self.run_name_ts,
                    "checkpoints",
                    "best_model.pt",
                )
                self.save_checkpoint(
                    epoch, filename=filepath
                )  # Example filename with validation loss
                print("$$$ New best checkpoint $$$ ")
            else:
                if epoch - self.best_val_epoch >= self.patience:
                    self.stop_early = True

            return (
                val_loss,
                max_pe,
                total_max_pe_1m,
                total_mean_pe / len(flight_start_rec_indices),
                total_mean_pe_1m / len(flight_start_rec_indices),
                avg_drift_overall,
                max_drift_overall,
            )

    def test(self, epoch=0):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_samples = 0

            self.test_data_y = None
            self.test_data_y_hat = None
            self.new_flight_started = None
            i = 0
            for test_data in tqdm(self.test_dataloader, desc=">> test", mininterval=5):
                X = test_data[0].to(self.device)
                Y = test_data[1].to(self.device)

                total_samples += X.shape[0]
                y_hat = self.model(X)

                loss = self.loss_fn(y_hat, Y[:, :2])
                total_loss += loss.item()

                if self.log_debug_data:
                    if self.test_data_y is None:
                        self.test_data_y = Y
                        self.test_data_y_hat = y_hat
                        self.new_flight_started = test_data[2].tolist()
                    else:
                        self.test_data_y = torch.cat((self.test_data_y, Y), dim=0)
                        self.test_data_y_hat = torch.cat(
                            (self.test_data_y_hat, y_hat), dim=0
                        )
                        self.new_flight_started.extend(test_data[2].tolist())

            # Calculate test metrics
            test_loss = total_loss / total_samples if total_samples > 0 else 0

            # Calculate trajectory metrics
            max_pe = 0  # max position error across all flights
            total_mean_pe = 0  # mean position error across all flights
            total_mean_pe_1m = 0
            total_max_pe_1m = 0
            flight_counter = 0
            per_flight_drift = []
            all_drift_pcts = []

            # Process metrics for each flight in the test dataset
            # if self.new_flight_started:
            # Get indices where new flight started
            flight_start_rec_indices = [
                i for i, x in enumerate(self.new_flight_started) if x
            ]

            for i in range(len(flight_start_rec_indices)):
                start_index = flight_start_rec_indices[i]
                end_index = (
                    flight_start_rec_indices[i + 1]
                    if i < len(flight_start_rec_indices) - 1
                    else len(self.test_data_y)
                )
                delta_y = self.test_data_y[start_index:end_index]
                delta_y_hat = self.test_data_y_hat[start_index:end_index]
                y_cumsum = torch.cumsum(delta_y, dim=0)
                y_hat_cumsum = torch.cumsum(delta_y_hat, dim=0)
                y_hat_cumsum = torch.cat(
                    (y_hat_cumsum, y_cumsum[:, 2:3]), dim=1
                )  # add z axis to y_hat_cumsum
                distance = torch.sum(torch.linalg.norm(delta_y, dim=1))
                max_pe = max(max_pe, Utils.get_max_pe(y_cumsum, y_hat_cumsum))
                total_max_pe_1m = torch.max(
                    torch.tensor(total_max_pe_1m),
                    Utils.get_max_pe_norm(y_cumsum, y_hat_cumsum),
                )
                total_mean_pe += Utils.get_mean_pe(y_cumsum, y_hat_cumsum)
                total_mean_pe_1m += Utils.get_mean_pe_norm(y_cumsum, y_hat_cumsum)
                flight_pe = Utils.get_pe(y_cumsum, y_hat_cumsum)
                avg_drift, max_drift, avg_drift_pct, max_drift_pct, drift_pcts = (
                    Utils.get_drift_percentage(y_cumsum, y_hat_cumsum)
                )
                per_flight_drift.append((avg_drift_pct, max_drift_pct))
                all_drift_pcts.append(drift_pcts)
                if self.log_debug_data and epoch % 3 == 0:
                    self.log_combined_plot(
                        y_cumsum,
                        y_hat_cumsum,
                        distance,
                        drift_pcts,
                        flight_counter,
                        avg_drift,
                        max_drift,
                        avg_drift_pct,
                        max_drift_pct,
                        flight_pe,
                        "test/{}_combined.png".format(flight_counter),
                    )
                flight_counter += 1

            mean_pe = (
                total_mean_pe / len(flight_start_rec_indices)
                if flight_start_rec_indices
                else 0
            )
            mean_pe_1m = (
                total_mean_pe_1m / len(flight_start_rec_indices)
                if flight_start_rec_indices
                else 0
            )

            avg_drift_overall = sum(d[0] for d in per_flight_drift) / len(
                per_flight_drift
            )
            max_drift_overall = sum(d[1] for d in per_flight_drift) / len(
                per_flight_drift
            )

            return (
                test_loss,
                max_pe,
                total_max_pe_1m,
                mean_pe,
                mean_pe_1m,
                avg_drift_overall,
                max_drift_overall,
            )
