import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

def load_weather_data(file_path):
    """Load and process weather data from JSON file"""
    with open(file_path, 'r') as f:
        weather_data = json.load(f)
    
    # Extract hourly data
    hourly_data = weather_data['hourly']
    
    # Create DataFrame
    df = pd.DataFrame(hourly_data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    print(f"Weather data range: {df.index.min()} to {df.index.max()}")
    print(f"Weather data shape: {df.shape}")
    
    return df

def load_water_level_data(file_path):
    """Load and process water level data from JSON file"""
    with open(file_path, 'r') as f:
        water_data = json.load(f)
    
    # Extract time series data
    time_series = water_data['timeSeriesList']['timeSeries'][0]['timeSeriesData']
    
    # Create DataFrame
    data = []
    for entry in time_series:
        if entry['value'] != 'NaN' and entry['value'] is not None:
            data.append({
                'datetime': pd.to_datetime(entry['dt']),
                'water_level': float(entry['value']),
                'flag': entry.get('flag', '')
            })
    
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    
    print(f"Water level data range: {df.index.min()} to {df.index.max()}")
    print(f"Water level data shape: {df.shape}")
    
    return df

def resample_to_hourly(df, value_col='water_level', method='mean'):
    """Resample irregular data to hourly frequency"""
    if method == 'mean':
        resampled = df[value_col].resample('H').mean()
    elif method == 'median':
        resampled = df[value_col].resample('H').median()
    elif method == 'last':
        resampled = df[value_col].resample('H').last()
    else:
        raise ValueError("Method must be 'mean', 'median', or 'last'")
    
    return resampled

def align_datasets(weather_df, water_df):
    """Align weather and water level datasets to common time range"""
    # Find overlapping time period
    start_time = max(weather_df.index.min(), water_df.index.min())
    end_time = min(weather_df.index.max(), water_df.index.max())
    
    print(f"Overlapping period: {start_time} to {end_time}")
    
    # Calculate expected hours and days
    expected_hours = int((end_time - start_time).total_seconds() / 3600) + 1
    expected_days = expected_hours / 24
    print(f"Expected hours: {expected_hours}")
    print(f"Expected days: {expected_days:.1f}")
    
    # Create complete hourly index for the entire period
    hourly_index = pd.date_range(start=start_time, end=end_time, freq='H')
    
    # Reindex both datasets to complete hourly grid
    weather_aligned = weather_df.reindex(hourly_index)
    water_aligned = water_df.reindex(hourly_index)
    
    print(f"After alignment - Weather shape: {weather_aligned.shape}, Water shape: {water_aligned.shape}")
    
    return weather_aligned, water_aligned, hourly_index

def smooth_missing_data(df, method='advanced_interpolation'):
    """Advanced missing data handling with smoothing"""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    print(f"Missing data before processing: {df.isnull().sum().sum()} values")
    
    if method == 'advanced_interpolation':
        # Step 1: Forward fill for short gaps (< 3 hours)
        df_filled = df.copy()
        for col in df.columns:
            # Forward fill short gaps
            df_filled[col] = df_filled[col].fillna(method='ffill', limit=3)
            # Backward fill remaining short gaps
            df_filled[col] = df_filled[col].fillna(method='bfill', limit=3)
        
        # Step 2: Interpolate remaining gaps
        df_filled = df_filled.interpolate(method='time', limit_direction='both')
        
        # Step 3: For remaining NaN values, use seasonal patterns
        for col in df.columns:
            if df_filled[col].isnull().any():
                # Calculate hourly seasonal patterns
                hourly_pattern = df_filled[col].groupby(df_filled.index.hour).median()
                daily_pattern = df_filled[col].groupby(df_filled.index.dayofyear).median()
                
                # Fill remaining NaN with seasonal patterns
                for idx in df_filled[df_filled[col].isnull()].index:
                    hour = idx.hour
                    day_of_year = idx.dayofyear
                    
                    # Try hourly pattern first
                    if not pd.isna(hourly_pattern.iloc[hour]):
                        df_filled.loc[idx, col] = hourly_pattern.iloc[hour]
                    # Then daily pattern
                    elif not pd.isna(daily_pattern.iloc[day_of_year % len(daily_pattern)]):
                        df_filled.loc[idx, col] = daily_pattern.iloc[day_of_year % len(daily_pattern)]
                    # Finally, use overall median
                    else:
                        df_filled.loc[idx, col] = df_filled[col].median()
        
        # Step 4: Apply smoothing to reduce noise
        df_smoothed = df_filled.copy()
        for col in df.columns:
            # Rolling average with window of 3 hours
            df_smoothed[col] = df_filled[col].rolling(window=3, center=True, min_periods=1).mean()
        
        return df_smoothed
    
    elif method == 'knn_imputation':
        # Use KNN imputation for more sophisticated missing value handling
        imputer = KNNImputer(n_neighbors=5)
        
        # Add time features to help with imputation
        df_with_time = df.copy()
        df_with_time['hour'] = df_with_time.index.hour
        df_with_time['day_of_year'] = df_with_time.index.dayofyear
        df_with_time['month'] = df_with_time.index.month
        
        # Apply KNN imputation
        imputed_values = imputer.fit_transform(df_with_time)
        
        # Remove time features and return original columns
        df_imputed = pd.DataFrame(imputed_values[:, :-3], 
                                 index=df.index, 
                                 columns=df.columns)
        
        return df_imputed
    
    else:
        # Original method as fallback
        if method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'interpolate':
            df = df.interpolate(method='linear')
        elif method == 'drop':
            df = df.dropna()
        
        return df

def create_sequences(weather_data, water_data, historical_days=60, forecast_days=7):
    """Create sequences for DeepAR training with improved handling"""
    historical_hours = historical_days * 24
    forecast_hours = forecast_days * 24
    
    sequences = []
    
    # Ensure we have enough data
    min_length = historical_hours + forecast_hours
    if len(weather_data) < min_length:
        raise ValueError(f"Not enough data. Need at least {min_length} hours, got {len(weather_data)}")
    
    # Select relevant weather features
    weather_features = [
        'temperature_2m', 'precipitation', 'evapotranspiration', 
        'et0_fao_evapotranspiration', 'snowfall', 'snow_depth',
        'soil_moisture_0_to_1cm', 'soil_moisture_1_to_3cm', 
        'soil_moisture_3_to_9cm', 'soil_moisture_9_to_27cm',
        'soil_moisture_27_to_81cm', 'relative_humidity_2m'
    ]
    
    # Filter weather features that exist in the data
    available_features = [feat for feat in weather_features if feat in weather_data.columns]
    
    # Create sequences with sliding window
    step_size = 24  # Create sequences every 24 hours (daily)
    
    for i in range(0, len(weather_data) - min_length + 1, step_size):
        # Historical period (60 days)
        hist_start = i
        hist_end = i + historical_hours
        
        # Future weather period (7 days)
        future_start = hist_end
        future_end = hist_end + forecast_hours
        
        # Target water level period (7 days)
        target_start = hist_end
        target_end = hist_end + forecast_hours
        
        # Extract sequences
        hist_weather = weather_data[available_features].iloc[hist_start:hist_end].values
        
        # Handle water data - might be DataFrame or Series
        if isinstance(water_data, pd.DataFrame):
            hist_water = water_data.iloc[hist_start:hist_end, 0].values  # Get first column
            target_water = water_data.iloc[target_start:target_end, 0].values
        else:
            hist_water = water_data.iloc[hist_start:hist_end].values
            target_water = water_data.iloc[target_start:target_end].values
            
        future_weather = weather_data[available_features].iloc[future_start:future_end].values
        
        # Since we've already handled missing data, we should have complete sequences
        # But double-check for any remaining issues
        if (not np.isnan(hist_weather).any() and 
            not np.isnan(hist_water).any() and 
            not np.isnan(future_weather).any() and 
            not np.isnan(target_water).any()):
            
            sequences.append({
                'historical_weather': hist_weather,
                'historical_water': hist_water,
                'future_weather': future_weather,
                'target_water': target_water,
                'timestamp': weather_data.index[hist_start]
            })
    
    return sequences, available_features

def prepare_training_data(sequences):
    """Prepare data for DeepAR training"""
    X_hist_weather = np.array([seq['historical_weather'] for seq in sequences])
    X_hist_water = np.array([seq['historical_water'] for seq in sequences])
    X_future_weather = np.array([seq['future_weather'] for seq in sequences])
    y = np.array([seq['target_water'] for seq in sequences])
    
    # Reshape water data for concatenation
    X_hist_water = X_hist_water[..., np.newaxis]
    # Combine historical data
    X_historical = np.concatenate([X_hist_weather, X_hist_water], axis=2)
    
    return X_historical, X_future_weather, y

def plot_data_overview(weather_df, water_df, save_path='data_overview.png'):
    """Plot overview of the data"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Handle water_df - it might be a Series or DataFrame
    if isinstance(water_df, pd.DataFrame):
        water_values = water_df.iloc[:, 0].values  # Get first column values
        water_series = water_df.iloc[:, 0]  # Get first column as Series
    else:
        water_values = water_df.values
        water_series = water_df
    
    # Water level over time
    axes[0, 0].plot(water_df.index, water_values)
    axes[0, 0].set_title('Water Level Over Time')
    axes[0, 0].set_ylabel('Water Level (ft)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Water level distribution
    axes[0, 1].hist(water_values, bins=50, alpha=0.7)
    axes[0, 1].set_title('Water Level Distribution')
    axes[0, 1].set_xlabel('Water Level (ft)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Temperature over time
    if 'temperature_2m' in weather_df.columns:
        axes[1, 0].plot(weather_df.index, weather_df['temperature_2m'])
        axes[1, 0].set_title('Temperature Over Time')
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Precipitation over time
    if 'precipitation' in weather_df.columns:
        axes[1, 1].plot(weather_df.index, weather_df['precipitation'])
        axes[1, 1].set_title('Precipitation Over Time')
        axes[1, 1].set_ylabel('Precipitation (mm)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Correlation between weather and water level
    if 'temperature_2m' in weather_df.columns:
        # Resample to daily for correlation
        daily_weather = weather_df['temperature_2m'].resample('D').mean()
        daily_water = water_series.resample('D').mean()
        
        # Align for correlation
        common_dates = daily_weather.index.intersection(daily_water.index)
        if len(common_dates) > 0:
            axes[2, 0].scatter(daily_weather.loc[common_dates], daily_water.loc[common_dates], alpha=0.5)
            axes[2, 0].set_title('Temperature vs Water Level')
            axes[2, 0].set_xlabel('Temperature (°C)')
            axes[2, 0].set_ylabel('Water Level (ft)')
    
    # Seasonal patterns
    if len(water_df) > 24*30:  # At least 30 days of data
        water_df_copy = water_series.to_frame(name='water_level')
        water_df_copy['hour'] = water_df_copy.index.hour
        hourly_avg = water_df_copy.groupby('hour')['water_level'].mean()
        axes[2, 1].plot(hourly_avg.index, hourly_avg.values)
        axes[2, 1].set_title('Average Water Level by Hour')
        axes[2, 1].set_xlabel('Hour of Day')
        axes[2, 1].set_ylabel('Water Level (ft)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load data
    print("Loading weather data...")
    weather_df = load_weather_data('owf-tstool-ai-training/weather_data.json')
    
    print("Loading water level data...")
    water_df = load_water_level_data('owf-tstool-ai-training/water_data.json')
    
    # Resample water level data to hourly
    print("Resampling water level data to hourly...")
    water_hourly = resample_to_hourly(water_df, method='mean')
    
    # Align datasets
    print("Aligning datasets...")
    weather_aligned, water_aligned, common_index = align_datasets(weather_df, water_hourly)
    
    # Handle missing data with advanced smoothing
    print("Handling missing data with advanced smoothing...")
    weather_clean = smooth_missing_data(weather_aligned, method='advanced_interpolation')
    water_clean = smooth_missing_data(water_aligned, method='advanced_interpolation')
    
    # Verify we have complete data
    print(f"Missing data after processing - Weather: {weather_clean.isnull().sum().sum()}, Water: {water_clean.isnull().sum().sum()}")
    
    # Final shape should match expected hours
    print(f"Final aligned data shape - Weather: {weather_clean.shape}, Water: {water_clean.shape}")
    print(f"Data covers {len(weather_clean) / 24:.1f} days")
    
    # Plot data overview
    print("Plotting data overview...")
    plot_data_overview(weather_clean, water_clean)
    
    # Create sequences
    print("Creating sequences for training...")
    sequences, feature_names = create_sequences(weather_clean, water_clean)
    
    print(f"Created {len(sequences)} sequences")
    print(f"Weather features used: {feature_names}")
    
    # Prepare training data
    print("Preparing training data...")
    X_historical, X_future_weather, y = prepare_training_data(sequences)
    
    print(f"Training data shapes:")
    print(f"  Historical data: {X_historical.shape}")
    print(f"  Future weather: {X_future_weather.shape}")
    print(f"  Target water levels: {y.shape}")
    
    # Split data 80/20
    split_idx = int(0.8 * len(sequences))
    
    X_hist_train = X_historical[:split_idx]
    X_future_train = X_future_weather[:split_idx]
    y_train = y[:split_idx]
    
    X_hist_test = X_historical[split_idx:]
    X_future_test = X_future_weather[split_idx:]
    y_test = y[split_idx:]
    
    print(f"Training set: {X_hist_train.shape[0]} samples")
    print(f"Test set: {X_hist_test.shape[0]} samples")
    
    # Save processed data
    print("Saving processed data...")
    np.savez('processed_data.npz',
             X_hist_train=X_hist_train,
             X_future_train=X_future_train,
             y_train=y_train,
             X_hist_test=X_hist_test,
             X_future_test=X_future_test,
             y_test=y_test,
             feature_names=feature_names)
    
    # Save scalers for later use
    scaler_weather = StandardScaler()
    scaler_water = StandardScaler()
    
    # Fit scalers on training data
    weather_features_train = X_hist_train[:, :, :-1].reshape(-1, X_hist_train.shape[2]-1)
    water_features_train = X_hist_train[:, :, -1].reshape(-1, 1)
    
    scaler_weather.fit(weather_features_train)
    scaler_water.fit(water_features_train)
    
    # Save scalers
    import joblib
    joblib.dump(scaler_weather, 'weather_scaler.pkl')
    joblib.dump(scaler_water, 'water_scaler.pkl')
    
    print("Data preprocessing completed successfully!")
    print(f"Processed data saved to: processed_data.npz")
    print(f"Scalers saved to: weather_scaler.pkl, water_scaler.pkl")
    print(f"Final dataset includes {len(weather_clean) / 24:.1f} days of complete data")

if __name__ == "__main__":
    main()