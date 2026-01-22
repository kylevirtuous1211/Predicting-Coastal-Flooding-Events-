import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import math
import sys
import argparse
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from einops import rearrange
from jaxtyping import Float, Int
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import scipy.io
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# 1. Configuration (TimeRCDConfig)
# ==========================================

@dataclass
class TimeSeriesConfig:
    d_model: int = 512
    d_proj: int = 256
    patch_size: int = 4
    num_query_tokens: int = 1
    num_layers: int = 8
    num_heads: int = 8
    d_ff_dropout: float = 0.1
    use_rope: bool = True
    activation: str = "gelu"
    num_features: int = 1

@dataclass
class TimeRCDConfig:
    ts_config: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    batch_size: int = 3
    learning_rate: float = 1e-4
    num_epochs: int = 1000
    max_seq_len: int = 512
    dropout: float = 0.1
    accumulation_steps: int = 1
    weight_decay: float = 1e-5
    enable_ts_train: bool = False
    seed: int = 72
    
    def to_dict(self) -> Dict[str, any]:
        return {"ts_config": self.ts_config.__dict__}

default_config = TimeRCDConfig()

# ==========================================
# 2. Time Series Encoder (ts_encoder_bi_bias.py)
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.to(torch.float32).pow(2).mean(dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs

class BinaryAttentionBias(nn.Module):
    def __init__(self, num_heads: Int):
        super().__init__()
        self.num_heads = num_heads
        self.emd = nn.Embedding(2, num_heads)

    def forward(self, query_id, kv_id):
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        ind = ind.unsqueeze(1)
        weight = rearrange(self.emd.weight, "two num_heads -> two num_heads 1 1")
        bias = ~ind * weight[:1] + ind * weight[1:]
        return bias

class MultiheadAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, num_features):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        if num_features > 1:
            self.binary_attention_bias = BinaryAttentionBias(num_heads)

    def apply_rope(self, x, freqs):
        B, seq_len, embed_dim = x.shape
        x_ = x.view(B, seq_len, embed_dim // 2, 2)
        cos = freqs.cos().unsqueeze(0)
        sin = freqs.sin().unsqueeze(0)
        x_rot = torch.stack([x_[..., 0] * cos - x_[..., 1] * sin, x_[..., 0] * sin + x_[..., 1] * cos], dim=-1)
        return x_rot.view(B, seq_len, embed_dim)

    def forward(self, query, key, value, freqs, query_id=None, kv_id=None, attn_mask=None):
        B, T, C = query.shape
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        Q_rot = self.apply_rope(Q, freqs)
        K_rot = self.apply_rope(K, freqs)
        Q_rot = Q_rot.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K_rot = K_rot.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if attn_mask is not None:
             attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        else:
             attn_mask = None

        if query_id is not None and kv_id is not None:
            attn_bias = self.binary_attention_bias(query_id, kv_id)
            scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += attn_bias
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            y = torch.matmul(attn_weights, V)
        else:
            y = F.scaled_dot_product_attention(Q_rot, K_rot, V, attn_mask=attn_mask, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y

class LlamaMLP(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, dim_feedforward, bias=True)
        self.up_proj = nn.Linear(d_model, dim_feedforward, bias=True)
        self.down_proj = nn.Linear(dim_feedforward, d_model, bias=True)
        self.act_fn = F.gelu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class TransformerEncoderLayerWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", num_features=1):
        super().__init__()
        self.self_attn = MultiheadAttentionWithRoPE(d_model, nhead, num_features)
        self.dropout = nn.Dropout(dropout)
        self.input_norm = RMSNorm(d_model)
        self.output_norm = RMSNorm(d_model)
        self.mlp = LlamaMLP(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, freqs, src_id=None, attn_mask=None):
        residual = src
        src = self.input_norm(src)
        src = self.self_attn(src, src, src, freqs, src_id, src_id, attn_mask=attn_mask)
        src = src + residual
        residual = src
        src = self.output_norm(src)
        src = self.mlp(src)
        src = residual + self.dropout2(src)
        return src

class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, num_layers, num_features):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(d_model, nhead, dim_feedforward, dropout, activation, num_features) 
            for _ in range(num_layers)
        ])

    def forward(self, src, freqs, src_id=None, attn_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, freqs, src_id, attn_mask=attn_mask)
        return output

class TimeSeriesEncoder(nn.Module):
    def __init__(self, d_model=2048, d_proj=512, patch_size=32, num_layers=6, num_heads=8,
                 d_ff_dropout=0.1, max_total_tokens=8192, use_rope=True, num_features=1, activation="relu"):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_proj = d_proj
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff_dropout = d_ff_dropout
        self.max_total_tokens = max_total_tokens
        self.use_rope = use_rope
        self.num_features = num_features
        self.activation = activation
        self.embedding_layer = nn.Linear(patch_size, d_model)

        if use_rope:
            self.rope_embedder = RotaryEmbedding(d_model)
            self.transformer_encoder = CustomTransformerEncoder(d_model, num_heads, d_model * 4, d_ff_dropout, activation, num_layers, num_features)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4, d_ff_dropout, batch_first=True, activation=activation)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.projection_layer = nn.Linear(d_model, patch_size * d_proj)
        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'linear' in name:
                 nn.init.kaiming_uniform_(param, nonlinearity='relu' if self.activation == 'relu' else 'gelu')
            elif 'bias' in name:
                 nn.init.constant_(param, 0.0)

    def forward(self, time_series, mask):
        if time_series.dim() == 2: time_series = time_series.unsqueeze(-1)
        device = time_series.device
        B, seq_len, num_features = time_series.size()
        padded_length = math.ceil(seq_len / self.patch_size) * self.patch_size
        if padded_length > seq_len:
            pad_amount = padded_length - seq_len
            time_series = F.pad(time_series, (0, 0, 0, pad_amount), value=0)
            mask = F.pad(mask, (0, pad_amount), value=0)

        num_patches = padded_length // self.patch_size
        total_length = num_patches * num_features
        patches = time_series.view(B, num_patches, self.patch_size, num_features).permute(0, 3, 1, 2).contiguous()
        patches = patches.view(B, num_features * num_patches, self.patch_size)
        
        feature_id = torch.arange(num_features, device=device).repeat_interleave(num_patches).unsqueeze(0).expand(B, -1)
        embedded_patches = self.embedding_layer(patches)

        mask_view = mask.view(B, num_patches, self.patch_size)
        patch_mask = mask_view.sum(dim=-1) > 0
        full_mask = patch_mask.unsqueeze(1).expand(-1, num_features, -1).reshape(B, num_features * num_patches)

        freqs = self.rope_embedder(total_length).to(device) if self.use_rope else None
        
        if num_features > 1:
            output = self.transformer_encoder(embedded_patches, freqs=freqs, src_id=feature_id, attn_mask=full_mask)
        else:
            output = self.transformer_encoder(embedded_patches, freqs=freqs, attn_mask=full_mask)

        patch_embeddings = output
        patch_proj = self.projection_layer(patch_embeddings)
        local_embeddings = patch_proj.view(B, num_features, num_patches, self.patch_size, self.d_proj)
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]
        return local_embeddings

# ==========================================
# 3. Time Series Pretrain Model (TimeRCD_pretrain_multi.py)
# ==========================================

class TimeSeriesPretrainModel(nn.Module):
    def __init__(self, config: TimeRCDConfig):
        super().__init__()
        self.config = config
        ts_config = config.ts_config
        self.ts_encoder = TimeSeriesEncoder(
            d_model=ts_config.d_model, d_proj=ts_config.d_proj, patch_size=ts_config.patch_size,
            num_layers=ts_config.num_layers, num_heads=ts_config.num_heads, d_ff_dropout=ts_config.d_ff_dropout,
            use_rope=ts_config.use_rope, num_features=ts_config.num_features, activation=ts_config.activation
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(ts_config.d_proj, ts_config.d_proj * 4), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(ts_config.d_proj * 4, ts_config.d_proj * 4), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(ts_config.d_proj * 4, 1)
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(ts_config.d_proj, ts_config.d_proj // 2), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(ts_config.d_proj // 2, 2)
        )

    def forward(self, time_series: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.ts_encoder(time_series, mask)

    def masked_reconstruction_loss(self, local_embeddings, original_time_series, mask):
        batch_size, seq_len, num_features = original_time_series.shape
        mask = mask.bool()
        reconstructed = self.reconstruction_head(local_embeddings).view(batch_size, seq_len, num_features)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_features)
        return F.mse_loss(reconstructed[mask_expanded], original_time_series[mask_expanded])

# ==========================================
# 4. Dataset (FloodDataset)
# ==========================================

class FloodDataset(Dataset):
    def __init__(self, data_path, split='train', context_len=168, pred_len=336):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.station_data = data[split]
        self.context_len = context_len
        self.pred_len = pred_len
        self.full_len = context_len + pred_len
        self.index_map = []
        for s_idx, station in enumerate(self.station_data):
            num_samples = len(station['X'])
            for i in range(num_samples):
                self.index_map.append((s_idx, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        s_idx, local_idx = self.index_map[idx]
        item = self.station_data[s_idx]
        X = item['X'][local_idx]
        Y = item['Y'][local_idx]
        full_seq = np.concatenate([X, Y])
        full_seq = torch.FloatTensor(full_seq).unsqueeze(-1)
        mask = torch.zeros(self.full_len, dtype=torch.bool)
        mask[self.context_len:] = True
        return {'time_series': full_seq, 'mask': mask, 'threshold': item['threshold'], 'station_name': item['name']}

# ==========================================
# 5. Preprocessing (preprocess_foundation.py)
# ==========================================

def preprocess_data(data_file_path):
    print("Preprocessing data...")
    # (Simplified for brevity as data is pickled. If raw data processing is needed, insert full logic here)
    # Since we use foundation_data.pkl, we assume it's created.
    pass

# ==========================================
# 6. Main Execution
# ==========================================

# ==========================================
# 6. Ingestion Support (CSV & Metadata)
# ==========================================

class IngestionDataset(Dataset):
    def __init__(self, train_csv, test_csv, test_index_csv, metadata_path, context_len=168, pred_len=336):
        self.context_len = context_len
        self.pred_len = pred_len
        self.full_len = context_len + pred_len

        # Load Metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Load Data
        print("Loading CSVs...")
        train_df = pd.read_csv(train_csv, parse_dates=['time'])
        test_df = pd.read_csv(test_csv, parse_dates=['time'])
        self.test_index = pd.read_csv(test_index_csv)

        # Combine and Sort
        self.full_df = pd.concat([train_df, test_df], ignore_index=True)
        self.full_df = self.full_df.sort_values(['station_name', 'time']).reset_index(drop=True)
        
        # Create Station Maps
        self.station_data = {}
        for name, group in self.full_df.groupby('station_name'):
            # Reindex to hourly to handle gaps (though ingestion fills gaps usually, but let's be safe)
            group = group.set_index('time').sort_index()
            # We assume data is hourly.
            # Normalize
            meta = self.metadata.get(name)
            if meta:
                mean = meta['mean'] # We normalized by (val - thresh) / std in training
                std = meta['std']
                thresh = meta['threshold'] # This is the original Station Threshold
                
                # IMPORTANT: TimeRCD preprocessing used (val - thresh) / std
                vals = group['sea_level'].values
                # Handle NaNs if any (interpolate)
                vals = pd.Series(vals).interpolate(limit=24).fillna(method='bfill').fillna(method='ffill').values
                
                norm_vals = (vals - thresh) / std
                
                # Store normalized values and index
                self.station_data[name] = {
                    'values': norm_vals,
                    'times': group.index
                }
            else:
                print(f"Warning: No metadata for {name}")

    def __len__(self):
        return len(self.test_index)
    
    def __getitem__(self, idx):
        row = self.test_index.iloc[idx]
        station_name = row['station_name']
        hist_start = pd.to_datetime(row['hist_start'])
        future_end = pd.to_datetime(row['future_end'])
        
        # We need History + Future window
        # In test_index, hist_start to hist_end is 7 days.
        # future_start to future_end is 14 days.
        # The query asks for a prediction for future window given history.
        # But TimeRCD takes the concatenated sequence.
        # We need to find the data corresponding to this range.
        
        data_info = self.station_data.get(station_name)
        if not data_info:
            return None # Should not happen

        times = data_info['times']
        values = data_info['values']
        
        # Find start index
        # We need 7 days history. 
        # The provided 'hist_start' and 'hist_end' cover the 7 days.
        
        # Search for hist_start
        start_search = times.searchsorted(hist_start)
        
        # We need 168 hours of history + 336 hours of future (masked)
        # Total 504 hours.
        # The timestamps for the 504 hours starting from hist_start:
        
        # Actually, let's verify exact timestamps.
        # hist_start is a DATE. ingestion says: 
        # hist_start = date, hist_end = date (7 days later inclusive?).
        # pd.to_datetime(row['hist_start']) gives 00:00:00 on that day.
        
        # We need to grab 168 hours starting from hist_start 00:00.
        # Then we append 336 zeros (placeholders) for the future.
        
        start_idx = start_search
        
        if start_idx + self.context_len > len(values):
            # Pad if history is missing (should verify data availability)
             hist_seq = values[start_idx:]
             pad_len = self.context_len - len(hist_seq)
             hist_seq = np.pad(hist_seq, (0, pad_len))
        else:
            hist_seq = values[start_idx : start_idx + self.context_len]
            
        # Future (Target) - we don't have it (or we masked it for prediction)
        future_seq = np.zeros(self.pred_len) 
        
        full_seq = np.concatenate([hist_seq, future_seq])
        full_seq = torch.FloatTensor(full_seq).unsqueeze(-1)
        
        mask = torch.zeros(self.full_len, dtype=torch.bool)
        mask[self.context_len:] = True # Mask future
        
        return {
            'time_series': full_seq,
            'mask': mask,
            'id': row['id'],
            'station_name': station_name
        }

def ingestion_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for metadata
    metadata_path = "station_metadata.pkl"
    if not os.path.exists(metadata_path):
        # Fallback if not found (e.g. create dummy or fail)
        print("Metadata not found! Please ensure station_metadata.pkl is present.")
        return

    dataset = IngestionDataset(
        args.train_hourly, 
        args.test_hourly, 
        args.test_index, 
        metadata_path
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load Model (Same logic)
    # We assume 'model.pkl' is in the current directory (submission root)
    checkpoint_path = "model.pkl"
    
    config = TimeRCDConfig()
    config.ts_config.num_features = 1
    config.ts_config.d_model = 512
    config.ts_config.patch_size = 16
    model = TimeSeriesPretrainModel(config).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): new_state_dict[k[7:]] = v
            else: new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("Model checkpoint model.pkl not found!")
        return

    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            if batch is None: continue
            
            time_series = batch['time_series'].to(device)
            mask = batch['mask'].to(device)
            attention_mask = torch.ones((time_series.size(0), time_series.size(1)), dtype=torch.bool).to(device)
            ids = batch['id']
            # station_names = batch['station_name']
            
            embeddings = model(time_series, attention_mask)
            reconstructed = model.reconstruction_head(embeddings).view(time_series.size(0), time_series.size(1), 1)
            
            # Reconstruction is Normalized Sea Level
            # Flood if Value > Threshold (which is 0.0 in normalized space)
            
            for i in range(len(ids)):
                future_mask = mask[i].bool()
                future_preds_norm = reconstructed[i][future_mask]
                
                # Binarize: Any hour > 0.0 => Flood
                flood_prob = (future_preds_norm > 0).float().max().item() # Max probability proxy? Or just binary?
                # In binary classification for this challenge: 1 for flood, 0 for no flood.
                # If we want probability, we might not have it directly from regression output.
                # But we can say 1.0 if any > 0 else 0.0
                
                label = 1 if (future_preds_norm > 0).any() else 0
                
                # results.append({'id': ids[i].item(), 'label': label})
                # Ingestion expects 'y_prob' or 'label'.
                results.append({'id': ids[i].item(), 'label': label})

    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv(args.predictions_out, index=False)
    print(f"Predictions saved to {args.predictions_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Standard modes from before
    parser.add_argument("--mode", type=str, default="evaluate", choices=["train", "evaluate"])
    parser.add_argument("--data", type=str, default="foundation_data.pkl")
    parser.add_argument("--model", type=str, default="model.pkl")
    
    # Ingestion arguments
    parser.add_argument("--train_hourly", type=str)
    parser.add_argument("--test_hourly", type=str)
    parser.add_argument("--test_index", type=str)
    parser.add_argument("--predictions_out", type=str)
    
    args = parser.parse_args()
    
    # Detect if running in ingestion mode
    if args.train_hourly and args.test_index:
        ingestion_predict(args)
    elif args.mode == "train":
        train(args.data, args.model)
    else:
        evaluate(args.data, args.model)