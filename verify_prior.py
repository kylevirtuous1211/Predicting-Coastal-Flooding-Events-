import torch
from model_prior_token import TimeRCDWithPrior, TimeRCDConfig
import traceback

def test_model():
    print("Initializing TimeRCDConfig...")
    config = TimeRCDConfig()
    print("Initializing TimeRCDWithPrior...")
    try:
        model = TimeRCDWithPrior(config, context_len=10) # 10 for quick test
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        traceback.print_exc()
        return

    B, T, C = 2, 50, 1
    time_series = torch.randn(B, T, C)
    mask = torch.ones(B, T)
    
    print("Running forward pass...")
    try:
        output = model(time_series, mask)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
        
        # Check output shape
        # output is local_embeddings: (B, T, 1, d_proj) or something like that?
        # model_prior_token.py line 292: 
        # local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]
        # So shape should be (B, seq_len, num_features, d_proj)
        
        expected_shape = (B, T, config.ts_config.num_features, config.ts_config.d_proj)
        if output.shape == expected_shape:
            print(f"Shape matches expected: {expected_shape}")
        else:
            print(f"Shape Mismatch! Expected {expected_shape}, got {output.shape}")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_model()
