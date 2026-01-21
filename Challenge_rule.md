s - Scientific Out-of-Distribution Challenge
Organized by: jona1 (jona1@umbc.edu)
Current Phase Ends: 2026年2月23日 下午7:59 [GMT+8]
Current server time: 2026年1月19日 下午2:03 [GMT+8]
Docker image: ytchou97/hdr-image:latest 
Training and Testing
Training Set:

Participants will be given hourly data from 1950 to 2020 for 9 coastal stations. This will be used for model training whereby participants can train their model on any 7-day historical interval and make predictions for the 14-day prediction interval following each historical time window given.
TRAINING_STATIONS = ['Annapolis','Atlantic_City','Charleston','Washington','Wilmington', 'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook']
Additional validation can be varied across select coastal stations (out of 9) to improve outcomes for modelling out-of-distribution across space.
Testing Set:

Participants will be given hourly data from 1950 to 2020 for 3 coastal stations, which will be used strictly for testing and evaluating out-of-distribution.
TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']
To test their models, participants have been given specific 15 historical time intervals (7-days per interval) as seed time intervals. These seed intervals have been selected between 1950 to 2020 and are not specific to any particular coastal station from the total of 12 coastal stations provided.
Participants also have the alternative to test their models using ANY user-defined 7-day historical time intervals to improve outcomes for modelling out-of-distribution across time.
Evaluation Metrics
This competition allows you to submit your developed algorithm, which will be run on the development and the final test dataset through CodaBench.

Expected Outputs: For EACH 14-day prediction time interval (following each 7-day historical time interval/seed window), your algorithm needs to generate the following outputs:

Evaluation Metrics: Models will be evaluated based on:

F1-Score
Accuracy
Matthews Correlation Coefficient (MCC)
Model Testing using Hidden Dataset
Hidden Test Dataset: Each participant's FINAL model will be evaluated on the hidden test dataset during the final phase. This test dataset comprises of FOUR (4) hidden coastal stations with hourly sea level measurements for the time period of 1950 to 2020. Each hidden station's file format is similar to those in the first training and test dataset that is accessible to the participants.

Model Expectations

The participant is expected to develop a global model that should predict across all stations. Therefore, the prediction should take into account 2520 entries (12 stations X 15 historical windows X 14 prediction windows), where 9 are for training and 3 for out-of-distribution testing.
Please note that the input to the model will be a 7-day window, but the participant is expected to predict for the next succeeding 14-day window period.
The dataset will be in hourly format, but the participant should consider developing the model on a daily interval rather than an hourly interval.
This will be treated as a binary prediction model. Please note that predictions will be done on a binary scale [1 -> Flooding, 0 -> Non-flooding]. We assume that a flooding day has at least ONE (1) hour of a flooding event. So given 24 hours in day D1, if any hour hx where x {1,… ,24} has a flooding event, then the day D1 is marked as flooding.
Evaluation should be based on per entry in the global model, hence 2520 entries derived from (12 stations X 15 historical windows X 14 prediction windows) where 9 are for training and 3 for out-of-distribution testing.
The output should consist of a Confusion Matrix indicating predicted flooding days.
The output should include the following evaluation metrics: Accuracy, F1-Score & Matthews Correlation Coefficient (MCC). Please note that predictions will be done on a binary scale [1 -> Flooding, 0 -> Non-flooding].
During the final phase models will be further tested using a hidden dataset containing 840 entries (4 stations X 15 historical windows X 14 prediction windows). This will be conducted 'secretly' by the challenge organizers.