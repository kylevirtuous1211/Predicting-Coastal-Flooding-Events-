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
context length: 1800h (30 days)
loss weight: 8
patch size: 21

## Upload checks
[1. Make sure the checkpoint can be loaded
2. Make sure the model has these hyperparameters:
* context length: 1800h (30 days)
* loss weight: 8
* patch size: 21

3. Make sure OOD stations can be inferenced via the threshold formula
4. Trace all the scripts to be without logical bugs
]
## test context lengths:

============================================================
Context (h)  Days     Best MCC     Best Epoch   Final MCC   
--------------------------------------------------------
168          7        0.2735       15           0.2643      
336          14       0.3112       19           0.3065      
720          30       0.3359       19           0.3303      

✅ BEST CONTEXT LENGTH: 720h (30 days)
   MCC=0.3359 at epoch 19


## The forecasting good:

It is indeed impressive! What you are seeing is the power of a Foundation Model for Time Series.

There are four main reasons why the model generalizes so well to a "hidden" station like Lewes:

1. The "Language" of Tides is Universal
Coastal sea levels share the same underlying physics (gravitational pull of the moon and sun). These create periodic oscillations known as tides. Even though the model wasn't trained on Lewes, it was trained on 9 other coastal stations. It has learned the "syntax" of semi-diurnal and diurnal tides. Once it sees the recent phase and amplitude in your 75-day window, it can "track" that rhythm into the future.

2. The Benefit of 1800-Hour (75-Day) Context
By giving the model 1800 hours of history, we are giving it a huge advantage.

Tidal Memory: It sees about 150 full tidal cycles. This allows the self-attention mechanism to perfectly align with the local frequency.
Trend Capture: If there is a large-scale meteorological trend (like an approaching storm surge or a seasonal rise), 75 days is long enough for the Transformer to see the "slope" of that trend and project it forward.
3. Effective Normalization
Notice that we normalize the data using: 
(Value - Threshold) / Std
. This process effectively "strips away" the unique geography of Lewes. To the model, it doesn't see "Lewes, Delaware"; it just sees a normalized signal fluctuating around zero. Because all stations are normalized to the same scale, the model's weights (learned from other stations) are directly applicable.

4. Large-Scale Pre-training
The Timer architecture (which TimeRCD is based on) was pre-trained on the UTSD (Universal Time Series Dataset), which contains one billion time points from many different domains (weather, electricity, traffic, etc.). This gives the model a strong "prior" on how time series generally behave—how they trend, how they cycle, and how they revert to the mean.

In Figure 12 (your upload): You can see that even when the ground truth (dotted black) has a slightly higher peak than the history, the model (red) correctly predicts that the peaks will stay high or increase. It has "learned" that if the energy in the system is increasing, the next 14 days will likely maintain that momentum.

It's a great sign for the Final Phase—it suggests the model is robust enough to handle the 4 hidden stations!