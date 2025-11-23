import torch
import torch.nn as nn


# ------------------------------------------------------------
# Minimal Transformer Encoder Layer WITHOUT LayerNorm
# ------------------------------------------------------------
class SimpleTransformerLayer(nn.Module):
    def __init__(self, d_model=1000, nhead=10, dim_feedforward=2048):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # No dropout, no LayerNorm
        self.act = nn.ReLU()

    def forward(self, x, attn_mask):
        # ---- Self Attention ----
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + attn_out  # residual

        # ---- Feed-forward ----
        ff = self.linear2(self.act(self.linear1(x)))
        x = x + ff  # residual

        return x


# ------------------------------------------------------------
# Decoder-Only Transformer (GPT-style)
# ------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        in_dim=11,
        model_dim=1000,
        num_layers=3,
        dim_feedforward=2048,
        nhead=10,
        out_dim=2,
    ):
        super().__init__()

        self.model_dim = model_dim

        # project sensor input → transformer dimension
        self.input_proj = nn.Linear(in_dim, model_dim)

        # build N layers
        self.layers = nn.ModuleList(
            [
                SimpleTransformerLayer(
                    d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward
                )
                for _ in range(num_layers)
            ]
        )

        # final output head (predict 2 values)
        self.fc = nn.Linear(model_dim, out_dim)

    def _causal_mask(self, T, device):
        """
        Returns a [T, T] causal mask with -inf above diagonal.
        """
        mask = torch.triu(torch.ones(T, T, device=device) * float("-inf"), diagonal=1)
        return mask

    def forward(self, x):
        """
        x: [B, T, in_dim]
        returns last-step prediction: [B, out_dim]
        """
        B, T, _ = x.size()

        # input embedding
        x = self.input_proj(x)  # → [B, T, model_dim]

        # causal mask
        mask = self._causal_mask(T, x.device)

        # transformer stack
        for layer in self.layers:
            x = layer(x, mask)  # [B, T, model_dim]

        # Get only last timestep (like LSTM last hidden state)
        last = x[:, -1, :]  # [B, model_dim]

        # Final prediction
        return self.fc(last)  # [B, out_dim]
