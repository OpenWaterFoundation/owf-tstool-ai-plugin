# OWF TSTool AI Training
The OWF TSTool AI Training folder contains the code and resources for preprocessing the data and training a custom DeepAR LSTM model for time series forecasting of water levels. This model is designed to predict future water levels and streamflow for flood warnings and to provide insights into water level trends.

| Script         | Description                                                                        |
|----------------|------------------------------------------------------------------------------------|
| train.py       | Script to train the DeepAR model                                                   |
| preproccing.py | Script to preprocces the data and convert it from two json files into usable files |

# Technical Explanation
The training script uses PyTorch to define and train a custom LSTM model with attention mechanisms. The model is trained on historical water level data and weather conditions to predict future water levels. For each step, the model takes in 60 days of historical weather and water level data + 7 days of future weather data to predict 7 days of future water level. The preprocessing script loads the data, resamples it, and prepares it for training.