from enum import Enum
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import torch
import pymap3d as pm
from datetime import datetime
from torch.optim.lr_scheduler import _LRScheduler
import math


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs=5, total_epochs=50, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_lr = [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
            return warmup_lr
        else:
            epoch_since_warmup = self.last_epoch - self.warmup_epochs
            cos_anneal_lr = [
                base_lr
                * 0.5
                * (
                    1
                    + math.cos(
                        math.pi
                        * epoch_since_warmup
                        / (self.total_epochs - self.warmup_epochs)
                    )
                )
                for base_lr in self.base_lrs
            ]
            return cos_anneal_lr

    def step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        super().step()


class LossAdaptiveLR(_LRScheduler):
    def __init__(
        self, optimizer, lr_list, patience=10, change_lr_patience=7, last_epoch=-1
    ):
        self.lr_list = lr_list
        self.patience = patience
        self.best_val_loss = float("inf")
        self.best_val_epoch = 0
        self.current_lr_index = 0
        self.lr_change_epoch = -1
        self.change_lr_patience = change_lr_patience
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if (
            self.last_epoch - self.best_val_epoch > self.patience
            and self.last_epoch - self.lr_change_epoch > self.change_lr_patience
        ):
            if self.current_lr_index + 1 < len(self.lr_list):
                self.lr_change_epoch = self.last_epoch
                self.current_lr_index += 1
        return [self.lr_list[self.current_lr_index] for _ in self.base_lrs]

    def step(self, val_loss=None):
        if val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_epoch = self.last_epoch
        super().step()


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch <= 50:
            return [0.0005 for _ in self.optimizer.param_groups]
        elif 51 <= epoch <= 100:
            return [0.00025 for _ in self.optimizer.param_groups]
        elif 101 <= epoch <= 150:
            return [0.00005 for _ in self.optimizer.param_groups]
        else:
            return [0.000005 for _ in self.optimizer.param_groups]


class Utils:
    """Collection of utility functions for data handling, plotting, and metrics."""

    @staticmethod
    def print_current_time(desc):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"{desc} =", current_time)

    @staticmethod
    def get_split(split_name, prefix_path=""):
        with open(os.path.join("data/splits", split_name, "train_dirs.txt"), "r") as f:
            train_dirs = [os.path.join(prefix_path, dir.strip() + ".csv") for dir in f]

        with open(os.path.join("data/splits", split_name, "test_dirs.txt"), "r") as f:
            test_dirs = [os.path.join(prefix_path, dir.strip() + ".csv") for dir in f]

        with open(os.path.join("data/splits", split_name, "val_dirs.txt"), "r") as f:
            val_dirs = [os.path.join(prefix_path, dir.strip() + ".csv") for dir in f]

        return train_dirs, test_dirs, val_dirs

    @staticmethod
    def create_splits(
        train_test_ulg_file_names_txt,
        val_ulg_file_names_txt,
        train_test_ulg_files_root_dir,
        val_ulg_files_root_dir,
        split_name,
        train_split=0.8,
    ):
        with open(train_test_ulg_file_names_txt, "r") as f:
            train_test_ulg_file_names = [line.strip() for line in f]

        with open(val_ulg_file_names_txt, "r") as f:
            val_ulg_file_names = [line.strip() for line in f]

        train_test_dirs = []
        val_dirs = []

        for root, dirs, _ in os.walk(train_test_ulg_files_root_dir):
            for dir in dirs:
                if dir.endswith(".ulg") and dir in train_test_ulg_file_names:
                    train_test_dirs.append(os.path.join(root, dir))

        for root, dirs, _ in os.walk(val_ulg_files_root_dir):
            for dir in dirs:
                if dir.endswith(".ulg") and dir in val_ulg_file_names:
                    val_dirs.append(os.path.join(root, dir))

        np.random.shuffle(train_test_dirs)
        split_idx = int(len(train_test_dirs) * train_split)
        train_dirs = train_test_dirs[:split_idx]
        test_dirs = train_test_dirs[split_idx:]

        split_dir = os.path.join("data/splits", split_name)
        os.makedirs(split_dir, exist_ok=True)

        with open(os.path.join(split_dir, "train_dirs.txt"), "w") as f:
            f.writelines([d + "\n" for d in train_dirs])

        with open(os.path.join(split_dir, "test_dirs.txt"), "w") as f:
            f.writelines([d + "\n" for d in test_dirs])

        with open(os.path.join(split_dir, "val_dirs.txt"), "w") as f:
            f.writelines([d + "\n" for d in val_dirs])

        return train_dirs, test_dirs, val_dirs

    @staticmethod
    def print_gpu_info():
        print("\n========================== GPU INFO ==========================")
        os.system("nvcc --version")
        print("--------------------------------------------------------------")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        torch.set_default_device(device)
        print("Is CUDA available: ", torch.cuda.is_available())
        print("N_GPU: ", n_gpu)
        if n_gpu > 0:
            print("GPU: ", torch.cuda.get_device_name(0))
        print("Torch default device: ", torch.get_default_device())
        print("--------------------------------------------------------------")
        print("Monitor GPU by running: watch -n 0.5 nvidia-smi")
        print("Monitor system resources with: free, top, htop")
        print("--------------------------------------------------------------")

    @staticmethod
    def save_csv(data, file_path):
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

    @staticmethod
    def get_ned(lat0, lon0, h0, lat, lon, h):
        e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
        return n, e, -u

    # ----------------- Dataset Preparation -----------------

    # ----------------- Metrics -----------------
    @staticmethod
    def get_max_pe(y, y_hat):
        return torch.max(torch.norm(y - y_hat, dim=1))

    @staticmethod
    def get_max_pe_norm(y, y_hat):
        max_pe_index = torch.argmax(torch.norm(y - y_hat, dim=1))
        dist = torch.sum(torch.norm(y[1 : max_pe_index + 1] - y[:max_pe_index], dim=1))
        max_pe = torch.max(torch.norm(y - y_hat, dim=1))
        return max_pe / dist

    @staticmethod
    def get_mean_pe(y, y_hat):
        return torch.mean(torch.norm(y - y_hat, dim=1))

    @staticmethod
    def get_pe(y, y_hat):
        return torch.norm(y - y_hat, dim=1)

    @staticmethod
    def get_drift_percentage(y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Compute per-flight Drift % metrics.

        Drift % is defined as:
        r_t = error_t / distance_travelled_t

        where:
        error_t = || y_pred[t] - y_true[t] ||
        distance_travelled_t = cumulative arc length of y_true up to t
        """

        # Use only x,y
        y_true_xy = y_true[:, :2]
        y_pred_xy = y_pred[:, :2]

        # Errors (absolute drift per step)
        errors = torch.norm(y_pred_xy - y_true_xy, dim=1)  # [T]

        # Stepwise distances in ground truth
        deltas = torch.norm(y_true_xy[1:] - y_true_xy[:-1], dim=1)  # [T-1]

        # Cumulative arc length
        dist_travelled = torch.cumsum(deltas, dim=0)  # [T-1]

        # Align errors with distances (skip t=0)
        errors_valid = errors[1:]  # [T-1]

        # Drift % per step
        drift_pcts = errors_valid / (dist_travelled + 1e-8) * 100.0

        # Absolute drift (not %)
        avg_drift = torch.mean(errors).item()
        max_drift = torch.max(errors).item()

        # Drift % stats
        avg_drift_pct = torch.mean(drift_pcts).item()
        max_drift_pct = torch.max(drift_pcts).item()

        return avg_drift, max_drift, avg_drift_pct, max_drift_pct, drift_pcts

    @staticmethod
    def get_mean_pe_norm(y, y_hat):
        dist = torch.sum(torch.norm(y[1:] - y[:-1], dim=1))
        mean_pe = torch.mean(torch.norm(y - y_hat, dim=1))
        return mean_pe / dist

    # ----------------- Plotting -----------------
    @staticmethod
    def plot_ts_in_2D(ts, label, legends, save_file_path, figsize=(10, 8)):
        fig, ax = plt.subplots(1, figsize=figsize)
        for i in range(len(ts)):
            ax.plot(range(len(ts[0])), ts[i], label=legends[i])
        ax.set_xlabel("Sample Index")
        ax.set_ylabel(label)
        ax.set_title("Predictions vs. Actual Values for " + label)
        ax.legend()
        ax.grid(True)
        ax.autoscale(enable=True, axis="x", tight=True)
        plt.tight_layout()
        if save_file_path is not None:
            plt.savefig(save_file_path, format="svg")

    @staticmethod
    def plot_two_trajectories_in_3D(y, y_hat, legends, save_file_path):
        y0, y1, y2 = zip(*y)
        y_hat0, y_hat1, y_hat2 = zip(*y_hat)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(y0, y1, y2, label=legends[0], linewidth=1.0)
        ax.plot(y_hat0, y_hat1, y_hat2, label=legends[1], linewidth=1.0)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Path")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_file_path, format="svg")


class ModelInput(Enum):
    (
        RAW_IMU,
        NUM_INT_POS_DIFF_1,
        NUM_INT_POS_DIFF_2,
        EKF_POS_DIFF,
        KF_POS_DIFF,
        RAW_DIFF_IMU,
        IMU_DIFF,
        IMU_DIFF_NORM,
        IMU_NORM,
    ) = range(9)


IN_DIM = {
    ModelInput.RAW_IMU: 6,
    ModelInput.NUM_INT_POS_DIFF_1: 3,
    ModelInput.NUM_INT_POS_DIFF_2: 3,
    ModelInput.EKF_POS_DIFF: 3,
    ModelInput.KF_POS_DIFF: 3,
    ModelInput.RAW_DIFF_IMU: 12,
    ModelInput.IMU_DIFF: 6,
    ModelInput.IMU_DIFF_NORM: 6,
    ModelInput.IMU_NORM: 6,
}


class ModelOutput(Enum):
    GPS, NED, SMOT_GPS, DELTA_SMOT_GPS, SMOT_NED, DELTA_SMOT_NED, DELTA_NED = range(7)
