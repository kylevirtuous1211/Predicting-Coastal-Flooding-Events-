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