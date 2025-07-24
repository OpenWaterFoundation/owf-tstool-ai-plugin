# Documentation for the Handover

# Technical Overview
## DeepAR LSTM
The DeepAR LSTM model is a time series forecasting model that uses Long Short-Term Memory (LSTM) networks to predict future values based on historical data. It is particularly effective for datasets with seasonal patterns and trends.
## DJL
DJL (Deep Java Library) is a Java-based framework for deep learning that provides a high-level API for building and training deep learning models. It supports various backends, including TensorFlow, PyTorch, and MXNet.
## PyTorch
PyTorch is the underlying deep learning framework used for building and training the DeepAR LSTM model. It is known for its dynamic computation graph and ease of use, making it a popular choice for research and production.

## Model
One timestep of the model is trained on 60 days of historical waterlevel and weather data and 7 days of forecast weather data.
The model uses weather data as input features and water level data as the target variable. The model is trained to predict the water level based on the historical weather data.
That means that for every timestep the model predicts the water level for the next 7 days.

# Using the model

## Data
https://open-meteo.com/en/docs
I used Open Meteo for the weather data. It has a weather Api for free and it is easy to use.
It has one for histrical data and one for forecast data.
It has many different parameters you can use.

I  used the following parameters for the historical data:

hourly_units:

- time

- temperature_2m

- precipitation

- evapotranspiration

- et0_fao_evapotranspiration

- snowfall

- snow_depth

- soil_moisture_0_to_1cm

- soil_moisture_1_to_3cm

- soil_moisture_3_to_9cm

- soil_moisture_9_to_27cm

- soil_moisture_27_to_81cm

- relative_humidity_2m

- precipitation_probability

The biggest possible time fram should be used for the historical data. The more data the better the model can learn from it.
The Longitude and Latitude of the location of the water level station should be used to get the weather data for that location.
So my api call looks like this:

https://api.open-meteo.com/v1/forecast?latitude=40.677868&longitude=-105.413367&hourly=temperature_2m,precipitation,evapotranspiration,et0_fao_evapotranspiration,snowfall,snow_depth,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm,soil_moisture_27_to_81cm,soil_moisture_9_to_27cm,relative_humidity_2m,precipitation_probability&past_days=60

This gets the 7 day hourly forcast and the past 60 days of historical data for the given latitude and longitude.

Then we just need the last 60 days of historical water level data.

## DJL Integration

Model is loaded using DJL’s Model API.

Input/output handled via DJL’s Translator interface.

Ensure the same data scaling and feature ordering is used as in training.

## Inference
### Input Requirements

Last 60 days of hourly weather + water level data

Next 7 days of hourly weather forecast

All values must be scaled using the same scalers used during training

### Output

Predicted hourly water level values for the next 7 days

Output format: array of timestamps + predicted values


# Training the model

## Data
For the weather data we use the same api provider but the historical data api.

https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=40.677868&longitude=-105.413367&start_date=2022-01-01&end_date=2025-07-12&hourly=temperature_2m,precipitation,evapotranspiration,et0_fao_evapotranspiration,snowfall,snow_depth,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm,soil_moisture_27_to_81cm,soil_moisture_9_to_27cm,relative_humidity_2m,precipitation_probability

For the water level data we use TSTool and save the data as a json file.

## Data preparation
in the owf-tstool-ai-training folder you will find the data preparation script.
It merges the weather data and the water level data into one file that is in the correct format for the model training.
It also splits the data into training and test data. So that the model can be evaluated on the test data after training. And not on data it has already seen.

It also creates scalers so when deploying the model we can scale the input data to the same range as the training data.


## Training
To train the model, we use the `train.py` script in the `owf-tstool-ai-training` folder. This script uses the prepared data and trains the DeepAR LSTM model using PyTorch.

## Converting the model
After training, the model is converted to a format that can be used with DJL. This is done using the `convert.py` which converts the PyTorch model to a format compatible with DJL.


