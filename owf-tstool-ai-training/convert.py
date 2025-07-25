import torch
import numpy as np

from training import ImprovedWaterLevelModel

# Load processed data to get feature_names and sequence length
data = np.load('./processed_data.npz', allow_pickle=True)
feature_names = data['feature_names'].tolist()
weather_features = len(feature_names)

# Infer sequence length from training data shape
X_hist_train = data['X_hist_train']
seq_len = X_hist_train.shape[1]

# Model parameters (match training_30.py)
hidden_size = 64
num_layers = 3
dropout = 0.25

# Instantiate model
model = ImprovedWaterLevelModel(
    weather_features=weather_features,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout
)

# Load checkpoint
checkpoint = torch.load('./best_water_level_model.pth', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create example input tensors
example_historical = torch.randn(1, seq_len, weather_features + 1)
example_future_weather = torch.randn(1, seq_len, weather_features)

# Trace and export TorchScript model
traced_model = torch.jit.trace(model, (example_historical, example_future_weather))
traced_model.save('../water_level_model.pt')

print("TorchScript model exported as '../water_level_model.pt'")
