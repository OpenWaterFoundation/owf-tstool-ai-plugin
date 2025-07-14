# OWF TSTool AI Plugin Documentation

## Background
The OWF TSTool AI Plugin is designed to implement Time series forcasting for Water levels using an Custom Deepar LSTM model. 
The goal is to predict future water levels flood warings and to provide insights into water level trends without the need for traditional modeling of a mathematical and physical model.
To get current future and historical Weather data, we used the open-meteo.com/ API.
We used TSTool to get the historical water level data from the database.
For the model training and data preprocessing, we used the Python programming language and the PyTorch library.
The plugin is writen in Java and the model ist deployed using DJL (Deep Java Library).
## Defining the Problem Domain
The problem domain is to predict future water levels based on historical data and weather conditions. For this to work we need to get the correct data for each location. So it would work as follows:
1. Select a water level sation that is of interest from the TSTool database.
2. Get the historical water level data for that station using TSTool.
3. Using the open-meteo.com/ API get the historical weather data for the same time period as the water level data.
4. Preprocess the data to make everything in the same hourly time steps and remove any missing values.
5. Use the preprocessed data to train the model.
6. Once the model is trained, it can be used to predict future water levels based on current and historical weather data and historical water levels.
## Reviewing and processing the input data
The input data is both in the json format and needs to be preprocced to have a regular hourly time step and to convert the data into .npy a format used by numpy and PyTorch to store the data as binary files for efficient loading.
## Configuring the model
In the Moment the model uses both 30 days of historrical water and weather data and 7 days of future weather data to predict the next 7 days of water levels. The model is a custom DeepAR LSTM model that is trained on the preprocessed data.
Other parameters that can be configured are the number of epochs, the batch size, and the learning rate. These parameters can be adjusted to improve the model's performance.
## Evaluating the Model
The model can be evaluated using various metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). These metrics can be calculated using the predictions made by the model on a test dataset that was not used during training. (We split the data into 80/20 split for training and testing.)
## Using the Model
The model can be used using the OWF TSTool AI Plugin. Which is written in Java und uses teh DJL (Deep Java Library).
