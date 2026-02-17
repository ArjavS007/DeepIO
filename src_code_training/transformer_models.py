# import torch
# import torch.nn as nn


# # ------------------------------------------------------------
# # Minimal Transformer Encoder Layer WITHOUT LayerNorm
# # ------------------------------------------------------------
# class SimpleTransformerLayer(nn.Module):
#     def __init__(self, d_model=1000, nhead=10, dim_feedforward=2048):
#         super().__init__()

#         self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         # No dropout, no LayerNorm
#         self.act = nn.ReLU()

#     def forward(self, x, attn_mask):
#         # ---- Self Attention ----
#         attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
#         x = x + attn_out  # residual

#         # ---- Feed-forward ----
#         ff = self.linear2(self.act(self.linear1(x)))
#         x = x + ff  # residual

#         return x


# # ------------------------------------------------------------
# # Decoder-Only Transformer (GPT-style)
# # ------------------------------------------------------------
# class DecoderOnlyTransformer(nn.Module):
#     def __init__(
#         self,
#         in_dim=11,
#         model_dim=1000,
#         num_layers=3,
#         dim_feedforward=2048,
#         nhead=10,
#         out_dim=2,
#     ):
#         super().__init__()

#         self.model_dim = model_dim

#         # project sensor input → transformer dimension
#         self.input_proj = nn.Linear(in_dim, model_dim)

#         # build N layers
#         self.layers = nn.ModuleList(
#             [
#                 SimpleTransformerLayer(
#                     d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward
#                 )
#                 for _ in range(num_layers)
#             ]
#         )

#         # final output head (predict 2 values)
#         self.fc = nn.Linear(model_dim, out_dim)

#     def _causal_mask(self, T, device):
#         """
#         Returns a [T, T] causal mask with -inf above diagonal.
#         """
#         mask = torch.triu(torch.ones(T, T, device=device) * float("-inf"), diagonal=1)
#         return mask

#     def forward(self, x):
#         """
#         x: [B, T, in_dim]
#         returns last-step prediction: [B, out_dim]
#         """
#         B, T, _ = x.size()

#         # input embedding
#         x = self.input_proj(x)  # → [B, T, model_dim]

#         # causal mask
#         mask = self._causal_mask(T, x.device)

#         # transformer stack
#         for layer in self.layers:
#             x = layer(x, mask)  # [B, T, model_dim]

#         # Get only last timestep (like LSTM last hidden state)
#         last = x[:, -1, :]  # [B, model_dim]

#         # Final prediction
#         return self.fc(last)  # [B, out_dim]

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        # Create a matrix of [max_len, d_model] representing positions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (won't be updated by optimizer)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class ImprovedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        
        # Feedforward block
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.act = nn.GELU() # Modern standard, smoother than ReLU

    def forward(self, x, attn_mask):
        # --- Pre-LN Architecture (more stable than Post-LN) ---
        
        # Self Attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = residual + self.dropout1(attn_out)

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = residual + self.dropout2(x)

        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, in_dim=11, model_dim=512, num_layers=4, nhead=8, dim_feedforward=2048, out_dim=2, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        
        self.layers = nn.ModuleList([
            ImprovedTransformerLayer(model_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Linear(model_dim, out_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Normal distribution with small std is standard for GPT/BERT
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def _causal_mask(self, T, device):
        mask = torch.triu(torch.ones(T, T, device=device) * float("-inf"), diagonal=1)
        return mask

    def forward(self, x):
        # x: [B, T, in_dim]
        B, T, _ = x.size()

        # 1. Project and add positions
        x = self.input_proj(x) 
        x = self.pos_encoder(x)

        # 2. Causal mask
        mask = self._causal_mask(T, x.device)

        # 3. Transformer stack
        for layer in self.layers:
            x = layer(x, mask)

        # 4. Final normalization (standard for Pre-LN)
        x = self.final_norm(x)

        # 5. Output head (last timestep)
        return self.fc(x[:, -1, :])