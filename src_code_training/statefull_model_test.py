import torch
from models import LSTMModelV3, LSTMStateConverter
from trainer import SimpleModelTrainerCpu
import os

if __name__ == "__main__":
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

    # 5. Prepare test data (same as in main_test_model.py)
    csv_files_dir = "/home/arjav.singh/Projects/DeepIO/Production_Data/SIDM/csvs"
    test_txt_file = "/home/arjav.singh/Projects/DeepIO/Production_Data/SIDM/src_code_training/test_data_traj/selected_test_traj.txt"
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

    trainer = SimpleModelTrainerCpu(
        stateful_model,
        test_flights,
        test_flights,
        test_flights,
        do_windowing=True,
        window_size=100,
        epochs=30,
        patience=5,
        run_name="GPU Test 4 - Stateful",
        lr=1e-5,
        lr_patience=7,
        last_lr_change_patience=5,
        use_lr_scheduler=True,
        pad_train_data=False,
        pad_val_data=False,
        clip_gradients=True,
        description="Testing the checkpoint 14",
        wandb_logging=True,
        batch_size=256,
        log_debug_data=True,
    )

    trainer.test(0)
