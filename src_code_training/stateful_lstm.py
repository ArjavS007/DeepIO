import torch.nn as nn
import torch


# -------------------------------
# Stateless LSTM Model
# -------------------------------
class LSTMModelV3(nn.Module):
    def __init__(
        self, in_dim=10, hidden_size=500, num_layers=1, output_size=3, dropout_prob=0.1
    ):
        super(LSTMModelV3, self).__init__()

        self.name = "LSTMModelV3"

        # Define stacked LSTM layers
        # Each LSTM layer receives input from the previous layer
        self.lstm_1 = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_4 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to map final hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass through each LSTM sequentially; default hidden states are zero
        x, _ = self.lstm_1(x)  # '_' ignores hidden/cell states
        x, _ = self.lstm_2(x)
        x, _ = self.lstm_3(x)
        x, _ = self.lstm_4(x)

        # Take the last time-step output from the last LSTM layer
        output = self.fc(x[:, -1, :])
        return output


# -------------------------------
# Stateful LSTM Model
# -------------------------------
class StatefulLSTMModelV3(nn.Module):
    def __init__(
        self,
        in_dim=10,
        hidden_size=500,
        num_layers=1,
        output_size=3,
        dropout_prob=0.1,
        device="cpu",
    ):
        super(StatefulLSTMModelV3, self).__init__()
        self.name = "StatefulLSTMModelV3"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # Define stacked LSTM layers (same as stateless model)
        self.lstm_1 = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_4 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # List to store hidden and cell states for each LSTM
        # This allows the model to "remember" state across batches
        self.hidden_states = [None] * 4  # Number of LSTM layers

    def reset_states(self, batch_size):
        """Reset hidden and cell states for all layers (usually at start of sequence)."""
        self.hidden_states = [
            (
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                    self.device
                ),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                    self.device
                ),
            )
            for _ in range(4)
        ]

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize states if they are None
        if self.hidden_states[0] is None:
            self.reset_states(batch_size)

        # Pass input through LSTM layers with their respective states
        # The output of each LSTM updates the hidden state for the next batch
        x, self.hidden_states[0] = self.lstm_1(x, self.hidden_states[0])
        x, self.hidden_states[1] = self.lstm_2(x, self.hidden_states[1])
        x, self.hidden_states[2] = self.lstm_3(x, self.hidden_states[2])
        x, self.hidden_states[3] = self.lstm_4(x, self.hidden_states[3])

        # Output from last time-step
        output = self.fc(x[:, -1, :])
        return output

    def detach_states(self):
        """Detach hidden states from the computational graph to prevent backpropagation through time indefinitely."""
        self.hidden_states = [(h.detach(), c.detach()) for (h, c) in self.hidden_states]


# -------------------------------
# Loading Pretrained Stateless Model
# -------------------------------
stateless_model = LSTMModelV3(in_dim=11, hidden_size=1400, output_size=2)
checkpoint_path = "/home/arjav.singh/Projects/DeepIO/Production_Data/SIDM/training_results/GPU Train 14-20250929-124430/checkpoints/best_model.pt"

# Load checkpoint (handles CPU/GPU automatically)
checkpoint = torch.load(
    checkpoint_path,
    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
stateless_model.load_state_dict(checkpoint["model_state_dict"])

# -------------------------------
# Convert Stateless to Stateful
# -------------------------------
stateful_model = StatefulLSTMModelV3(
    in_dim=stateless_model.lstm_1.input_size,
    hidden_size=stateless_model.lstm_1.hidden_size,
    num_layers=stateless_model.lstm_1.num_layers,
    output_size=stateless_model.fc.out_features,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Copy weights from stateless model to stateful model
stateful_model.load_state_dict(stateless_model.state_dict())

# -------------------------------
# Verify Conversion
# -------------------------------
# Experiment parameters (adjust as needed; use your original values if preferred)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 20
num_batches = 5
batch_size = 1
input_dim = 11  # Matches your stateless_model

# Move models to device
stateless_model.to(device)
stateful_model.to(device)

# Generate continuous input signal (sine wave + padding to input_dim)
t = torch.linspace(0, 10, steps=seq_len * num_batches)
signal = torch.sin(t).view(num_batches, seq_len, 1)  # Shape: (num_batches, seq_len, 1)
signal = torch.cat(
    [signal, torch.zeros(num_batches, seq_len, input_dim - 1)], dim=-1
).to(device)  # Shape: (num_batches, seq_len, input_dim)

# Full concatenated sequence for baseline
full_sequence = torch.cat([batch.unsqueeze(0) for batch in signal], dim=1).to(
    device
)  # Shape: (1, seq_len * num_batches, input_dim)

# 1. Baseline: Stateless on full sequence (single forward pass)
with torch.no_grad():
    y_stateless_full = stateless_model(full_sequence)

print("\nBaseline: Stateless output on full sequence")
print(y_stateless_full)

# 2. Sequential processing with stateless model (resets states each batch)
stateless_seq_outputs = []
with torch.no_grad():
    for i, batch in enumerate(signal):
        batch = batch.unsqueeze(0)  # Shape: (1, seq_len, input_dim)
        y_stateless = stateless_model(batch)
        stateless_seq_outputs.append(y_stateless)
        print(f"\nStateless sequential - Batch {i + 1} output:")
        print(y_stateless)

# Compare last stateless sequential output to full baseline (should differ, as it only sees the last batch)
diff_stateless_last_vs_full = (
    torch.abs(stateless_seq_outputs[-1] - y_stateless_full).max().item()
)
print(
    f"\nDifference (stateless seq last vs full baseline): {diff_stateless_last_vs_full} (expected: non-zero)"
)

# 3. Sequential processing with stateful model (carries states across batches)
stateful_seq_outputs = []
stateful_model.reset_states(batch_size=batch_size)
with torch.no_grad():
    for i, batch in enumerate(signal):
        batch = batch.unsqueeze(0)  # Shape: (1, seq_len, input_dim)
        y_stateful = stateful_model(batch)
        stateful_seq_outputs.append(y_stateful)
        # Optional: Call detach_states() here if this were training to truncate gradients

# Compare stateful final output to full baseline (should match exactly)
diff_stateful_final_vs_full = (
    torch.abs(stateful_seq_outputs[-1] - y_stateless_full).max().item()
)
print(
    f"\nDifference (stateful seq final vs full baseline): {diff_stateful_final_vs_full} (expected: ~0)"
)

# 4. More granular verification: Check stateful outputs against stateless on cumulative prefixes
# This confirms state carry-over at every step
print(
    "\nGranular prefix checks (stateful batch N vs stateless on batches 1-N concatenated):"
)
for n in range(1, num_batches + 1):
    prefix_sequence = torch.cat([batch.unsqueeze(0) for batch in signal[:n]], dim=1).to(
        device
    )  # Cumulative prefix
    with torch.no_grad():
        y_stateless_prefix = stateless_model(prefix_sequence)
    y_stateful_n = stateful_seq_outputs[n - 1]
    diff = torch.abs(y_stateful_n - y_stateless_prefix).max().item()
    print(f"Prefix up to batch {n}: Difference = {diff} (expected: ~0)")

# 5. Bonus: Repeat the same batch multiple times to show state dependence
# Reset states
stateful_model.reset_states(batch_size=batch_size)
repeated_batch = signal[0].unsqueeze(0)  # First batch repeated
print(
    "\nStateful on repeated identical batch (outputs should change due to state buildup):"
)
with torch.no_grad():
    for i in range(3):  # Repeat 3 times
        y_stateful_repeat = stateful_model(repeated_batch)
        print(f"Repeat {i + 1}: {y_stateful_repeat}")

print("\nStateless on repeated identical batch (outputs should be identical):")
with torch.no_grad():
    for i in range(3):
        y_stateless_repeat = stateless_model(repeated_batch)
        print(f"Repeat {i + 1}: {y_stateless_repeat}")
