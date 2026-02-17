import torch
import torch.nn as nn

# -------------------------
# RMSNorm (optional)
# -------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.scale


# -------------------------
# LSTMModelV6 (Sweep-ready)
# -------------------------
class LSTMModelV6(nn.Module):
    def __init__(
        self,
        in_dim=11,
        hidden_size=1024,
        output_size=2,
        num_blocks=6,
        lstm_layers=1,
        norm_type="layer",        # ["layer", "rms", "none"]
        norm_position="post",     # ["pre", "post"]
        residuals=True,
        residual_scale=1.0,
        dropout_lstm=0.0,
        dropout_output=0.1,
        init_type="orthogonal",   # ["orthogonal", "xavier", "default"]
    ):
        super().__init__()
        self.name = "LSTMModelV6"

        self.residuals = residuals
        self.residual_scale = residual_scale
        self.norm_position = norm_position
        self.norm_type = norm_type
        self.init_type = init_type

        # -------------------------
        # Input projection
        # -------------------------
        self.input_proj = nn.Linear(in_dim, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        # -------------------------
        # LSTM blocks
        # -------------------------
        self.lstm_blocks = nn.ModuleList([
            nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=lstm_layers,
                batch_first=True
            )
            for _ in range(num_blocks)
        ])

        # -------------------------
        # Norm blocks
        # -------------------------
        self.norm_blocks = nn.ModuleList([
            self._make_norm(hidden_size)
            for _ in range(num_blocks)
        ])

        # -------------------------
        # Dropouts
        # -------------------------
        self.block_dropout = nn.Dropout(dropout_lstm)
        self.output_dropout = nn.Dropout(dropout_output)

        # -------------------------
        # Output head
        # -------------------------
        self.fc = nn.Linear(hidden_size, output_size)

        # -------------------------
        # Weight initialization
        # -------------------------
        if self.init_type != "default":
            self.apply(self._init_weights)

    # -------------------------
    # Helpers
    # -------------------------
    def _make_norm(self, hidden_size):
        if self.norm_type == "layer":
            return nn.LayerNorm(hidden_size)
        elif self.norm_type == "rms":
            return RMSNorm(hidden_size)
        else:
            return nn.Identity()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.init_type == "orthogonal":
                nn.init.orthogonal_(module.weight)
            elif self.init_type == "xavier":
                nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    if self.init_type == "xavier":
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.xavier_uniform_(param)

                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)

                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Forget gate bias = 1.0
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x):
        """
        x: (B, T, in_dim)
        """

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)

        # LSTM stack
        for lstm, norm in zip(self.lstm_blocks, self.norm_blocks):
            residual = x

            if self.norm_position == "pre":
                x = norm(x)
                x, _ = lstm(x)
                x = self.block_dropout(x)
                if self.residuals:
                    x = x + self.residual_scale * residual

            else:  # post-norm
                x, _ = lstm(x)
                x = self.block_dropout(x)
                if self.residuals:
                    x = norm(x + self.residual_scale * residual)
                else:
                    x = norm(x)

        # Readout (last timestep)
        x = self.output_dropout(x)
        return self.fc(x[:, -1, :])


class LSTMModelV3(nn.Module):
    def __init__(
        self, in_dim=10, hidden_size=500, num_layers=1, output_size=3, dropout_prob=0.1
    ):
        super(LSTMModelV3, self).__init__()
        # Initialize hidden and cell states
        # self.h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        # self.c0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)

        self.name = "LSTMModelV3"
        self.lstm_1 = nn.LSTM(
            in_dim, hidden_size, num_layers, batch_first=True
        )
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        x, _ = self.lstm_3(x)
        output = self.fc(x[:, -1, :])
        return output

class StatefulLSTMModelV3(nn.Module):
    def __init__(
        self,
        in_dim=11,
        hidden_size=500,
        num_layers=1,
        output_size=2,
        dropout_prob=0.1,
        device="cpu",
    ):
        super(StatefulLSTMModelV3, self).__init__()
        self.name = "StatefulLSTMModelV3"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # Define stacked LSTMs
        self.lstm_1 = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_4 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize hidden states for each LSTM
        self.hidden_states = [None] * 4  # ! number of LSTM layers

    def reset_states(self, batch_size):
        """Reset all LSTM hidden and cell states"""
        self.hidden_states = [
            (
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                    self.device
                ),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
                    self.device
                ),
            )
            for _ in range(4)  # ! number of LSTM layers
        ]

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize states if None
        if self.hidden_states[0] is None:
            self.reset_states(batch_size)

        # Pass through each LSTM layer with its own state
        x, self.hidden_states[0] = self.lstm_1(x, self.hidden_states[0])
        x, self.hidden_states[1] = self.lstm_2(x, self.hidden_states[1])
        x, self.hidden_states[2] = self.lstm_3(x, self.hidden_states[2])
        x, self.hidden_states[3] = self.lstm_4(x, self.hidden_states[3])

        output = self.fc(x[:, -1, :])
        return output

    def detach_states(self):
        """Detach hidden states to prevent backprop through history"""
        self.hidden_states = [(h.detach(), c.detach()) for (h, c) in self.hidden_states]


class SwitchableLSTMModelV3(nn.Module):
    def __init__(
        self,
        in_dim=11,
        hidden_size=500,
        num_layers=1,
        output_size=2,
        dropout_prob=0.1,
        device="cpu",
    ):
        super(SwitchableLSTMModelV3, self).__init__()
        self.name = "SwitchableLSTMModelV3"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.stateful = False  # default: stateless mode

        # Define stacked LSTMs
        self.lstm_1 = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_4 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # Hidden states (only used in stateful mode)
        self.hidden_states = [None] * 4  # for 4 LSTM layers

    def set_stateful(self, mode: bool):
        """Enable or disable stateful mode."""
        self.stateful = mode
        if not mode:
            self.hidden_states = [None] * 4

    def reset_states(self, batch_size):
        """Reset all hidden and cell states for a given batch size."""
        self.hidden_states = [
            (
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
            )
            for _ in range(4)
        ]

    def detach_states(self):
        """Detach hidden states to prevent backprop through time."""
        if self.stateful:
            self.hidden_states = [
                (h.detach(), c.detach()) for (h, c) in self.hidden_states
            ]

    def forward(self, x):
        batch_size = x.size(0)

        if not self.stateful:
            # Stateless: always create fresh hidden states
            hidden_states = [
                (
                    torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                )
                for _ in range(4)
            ]
        else:
            # ! here I have modified the model to handle the cases where the dimesion of input does not match the batch size
            # Stateful: create or reset if batch size mismatches
            if self.hidden_states[0] is None or self.hidden_states[0][0].size(1) != batch_size:
                self.reset_states(batch_size)
            hidden_states = self.hidden_states

        # Forward pass through LSTM stack
        x, hidden_states[0] = self.lstm_1(x, hidden_states[0])
        x, hidden_states[1] = self.lstm_2(x, hidden_states[1])
        x, hidden_states[2] = self.lstm_3(x, hidden_states[2])
        x, hidden_states[3] = self.lstm_4(x, hidden_states[3])

        # Save hidden state if in stateful mode
        if self.stateful:
            self.hidden_states = hidden_states

        output = self.fc(x[:, -1, :])
        return output



class LSTMStateConverter:
    """
    Converts a trained stateless LSTM model into a stateful one,
    preserving all learned weights and configurations.
    """

    @staticmethod
    def convert_to_stateful(stateless_model, device="cpu", stateful_mode=True):
        """
        Convert a trained stateless LSTMModelV3 to a SwitchableLSTMModelV3.

        Args:
            stateless_model (nn.Module): Trained stateless LSTM model.
            device (str): Target device for the new model.
            stateful_mode (bool): Whether to enable stateful mode by default.

        Returns:
            SwitchableLSTMModelV3: A model with identical weights, but with stateful capability.
        """
        # Ensure model type
        if not hasattr(stateless_model, "lstm_1"):
            raise ValueError(
                "Input model does not seem to be an LSTMModelV3-like architecture."
            )

        # Import the class dynamically (assumes it's defined in the same file or namespace)
        model_cls = SwitchableLSTMModelV3

        # Create new model with same hyperparameters
        stateful_model = model_cls(
            in_dim=stateless_model.lstm_1.input_size,
            hidden_size=stateless_model.lstm_1.hidden_size,
            num_layers=stateless_model.lstm_1.num_layers,
            output_size=stateless_model.fc.out_features,
            device=device,
        )

        # Load weights
        stateful_model.load_state_dict(stateless_model.state_dict())

        # Enable stateful mode if required
        stateful_model.set_stateful(stateful_mode)
        stateful_model.to(device)

        return stateful_model
