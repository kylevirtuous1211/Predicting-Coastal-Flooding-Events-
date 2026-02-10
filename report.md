# Testing files structure
## test_hourly.csv - Station Data
time	station_name	sea_level
2020-01-01 00:00	Boston	1.234
2020-01-01 01:00	Boston	1.245

## test_index.csv - Hidden Dates
id	station_name	hist_start
0	Boston	2021-06-15 00:00
1	Boston	2021-06-16 00:00


## Prior token architecture 

┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PHASE 1                              │
│   ┌─────────────┐                                                       │
│   │  History    │──► FloodScout ──► Probability ──► BCE/Focal Loss      │
│   │  (168 hrs)  │         ▲                              ▲              │
│   └─────────────┘         │                              │              │
│                      Train Supervised              Ground Truth         │
│                                                    (did future flood?)  │
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PHASE 2                              │
│   ┌─────────────┐                                                       │
│   │  History    │──► FloodScout (FROZEN) ──► Prob                       │
│   │  (168 hrs)  │                              │                        │
│   └──────┬──────┘                              ▼                        │
│          │                         ┌───────────────────┐                │
│          │                         │ PriorEmbedding    │                │
│          │                         │ safe_token ◄──────┤                │
│          │                         │ risk_token ◄──────┤                │
│          │                         └────────┬──────────┘                │
│          │                                  │                           │
│          ▼                                  ▼                           │
│   ┌──────────────────────────────────────────────────┐                  │
│   │  [Prior_Token] + [Patch_1] + [Patch_2] + ...     │                  │
│   └──────────────────────────────────────────────────┘                  │
│                              │                                          │
│                              ▼                                          │
│                        TimeRCD Transformer                              │
│                              │                                          │
│                              ▼                                          │
│                    Reconstruction Loss (MSE)                            │
└─────────────────────────────────────────────────────────────────────────┘

## Best configuration:
context length: 720h (30 days)
loss weight: 8
patch size: 21

## test context lengths:

============================================================
Context (h)  Days     Best MCC     Best Epoch   Final MCC   
--------------------------------------------------------
168          7        0.2735       15           0.2643      
336          14       0.3112       19           0.3065      
720          30       0.3359       19           0.3303      

✅ BEST CONTEXT LENGTH: 720h (30 days)
   MCC=0.3359 at epoch 19