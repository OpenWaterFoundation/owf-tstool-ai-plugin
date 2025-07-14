import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class WaterLevelDataset(Dataset):
    """Custom dataset for water level forecasting"""
    def __init__(self, X_historical, X_future_weather, y, transform=None):
        self.X_historical = torch.FloatTensor(X_historical)
        self.X_future_weather = torch.FloatTensor(X_future_weather)
        self.y = torch.FloatTensor(y)
        self.transform = transform
        
    def __len__(self):
        return len(self.X_historical)
    
    def __getitem__(self, idx):
        return {
            'historical': self.X_historical[idx],
            'future_weather': self.X_future_weather[idx],
            'target': self.y[idx]
        }

class DeepARModel(nn.Module):
    """Improved DeepAR model with better stability"""
    def __init__(self, weather_features, hidden_size=64, num_layers=2, dropout=0.2):
        super(DeepARModel, self).__init__()
        
        self.weather_features = weather_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Historical encoder (weather + water level)
        self.historical_encoder = nn.LSTM(
            input_size=weather_features + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Future weather encoder
        self.future_encoder = nn.LSTM(
            input_size=weather_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,  # Reduced from 8
            dropout=dropout,
            batch_first=True
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_size + weather_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Add batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # Output layers with dropout
        self.dropout = nn.Dropout(dropout)
        self.fc_mean = nn.Linear(hidden_size, 1)
        self.fc_std = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights properly"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif param.dim() >= 2:  # Only apply xavier to 2D+ tensors
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.normal_(param, 0, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
    def forward(self, historical, future_weather):
        batch_size = historical.size(0)
        
        # Encode historical data
        hist_output, (hist_h, hist_c) = self.historical_encoder(historical)
        
        # Encode future weather
        future_output, _ = self.future_encoder(future_weather)
        
        # Apply attention
        attn_output, _ = self.attention(future_output, hist_output, hist_output)
        
        # Apply batch normalization and dropout
        attn_output = self.dropout(attn_output)
        
        # Prepare decoder input
        decoder_input = torch.cat([attn_output, future_weather], dim=2)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input, (hist_h, hist_c))
        
        # Apply dropout
        decoder_output = self.dropout(decoder_output)
        
        # Predict mean and std
        mean = self.fc_mean(decoder_output)
        
        # Improved std prediction with better bounds
        std_logits = self.fc_std(decoder_output)
        std = torch.clamp(torch.exp(std_logits), min=0.01, max=10.0)  # Better bounds
        
        return mean, std

def improved_gaussian_likelihood_loss(y_true, y_pred_mean, y_pred_std):
    """Improved Gaussian likelihood loss with better stability"""
    # Clamp std to prevent numerical issues
    y_pred_std = torch.clamp(y_pred_std, min=1e-3, max=10.0)
    
    # Compute loss components
    log_likelihood = -0.5 * torch.log(2 * np.pi * y_pred_std**2) - 0.5 * ((y_true - y_pred_mean)**2) / (y_pred_std**2)
    
    # Return negative log likelihood
    return -torch.mean(log_likelihood)

def train_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    """Train for one epoch with gradient clipping"""
    model.train()
    total_loss = 0
    batch_count = 0

    for batch in dataloader:
        historical = batch['historical'].to(device)
        future_weather = batch['future_weather'].to(device)
        target = batch['target'].to(device)
        target = target.unsqueeze(-1)

        optimizer.zero_grad()

        mean, std = model(historical, future_weather)
        loss = improved_gaussian_likelihood_loss(target, mean, std)

        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"  [train_epoch] Batch {batch_count}: Loss = {loss.item():.4f}")

    print(f"  [train_epoch] Epoch finished. Avg Loss = {total_loss / len(dataloader):.4f}")
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            historical = batch['historical'].to(device)
            future_weather = batch['future_weather'].to(device)
            target = batch['target'].to(device)
            target = target.unsqueeze(-1)

            mean, std = model(historical, future_weather)
            loss = improved_gaussian_likelihood_loss(target, mean, std)

            total_loss += loss.item()
            predictions.append(mean.cpu().numpy())
            targets.append(target.cpu().numpy())
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"  [validate_epoch] Batch {batch_count}: Loss = {loss.item():.4f}")

    print(f"  [validate_epoch] Epoch finished. Avg Loss = {total_loss / len(dataloader):.4f}")
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    return total_loss / len(dataloader), predictions, targets

def main():
    print("=== Improved DeepAR Water Level Training ===")
    # Load processed data
    print("Loading processed data...")
    data = np.load('processed_data.npz', allow_pickle=True)

    X_hist_train = data['X_hist_train']
    X_future_train = data['X_future_train']
    y_train = data['y_train']
    X_hist_test = data['X_hist_test']
    X_future_test = data['X_future_test']
    y_test = data['y_test']
    feature_names = data['feature_names'].tolist()

    print(f"Training data shapes:")
    print(f"  Historical: {X_hist_train.shape}")
    print(f"  Future weather: {X_future_train.shape}")
    print(f"  Target: {y_train.shape}")
    print(f"  Features: {feature_names}")

    # Normalize data
    print("Normalizing data...")
    scaler_weather = StandardScaler()
    scaler_water = StandardScaler()

    # Fit scalers
    print("Fitting weather and water level scalers...")
    weather_data_train = X_hist_train[:, :, :-1].reshape(-1, X_hist_train.shape[2]-1)
    water_data_train = X_hist_train[:, :, -1:].reshape(-1, 1)

    scaler_weather.fit(weather_data_train)
    scaler_water.fit(water_data_train)
    print("Scalers fitted.")

    # Transform training data
    print("Transforming training data...")
    X_hist_train_scaled = X_hist_train.copy()
    X_hist_train_scaled[:, :, :-1] = scaler_weather.transform(
        X_hist_train[:, :, :-1].reshape(-1, X_hist_train.shape[2]-1)
    ).reshape(X_hist_train.shape[0], X_hist_train.shape[1], -1)
    X_hist_train_scaled[:, :, -1:] = scaler_water.transform(
        X_hist_train[:, :, -1:].reshape(-1, 1)
    ).reshape(X_hist_train.shape[0], X_hist_train.shape[1], 1)

    X_future_train_scaled = scaler_weather.transform(
        X_future_train.reshape(-1, X_future_train.shape[2])
    ).reshape(X_future_train.shape)

    y_train_scaled = scaler_water.transform(
        y_train.reshape(-1, 1)
    ).reshape(y_train.shape)

    # Transform test data
    print("Transforming test data...")
    X_hist_test_scaled = X_hist_test.copy()
    X_hist_test_scaled[:, :, :-1] = scaler_weather.transform(
        X_hist_test[:, :, :-1].reshape(-1, X_hist_test.shape[2]-1)
    ).reshape(X_hist_test.shape[0], X_hist_test.shape[1], -1)
    X_hist_test_scaled[:, :, -1:] = scaler_water.transform(
        X_hist_test[:, :, -1:].reshape(-1, 1)
    ).reshape(X_hist_test.shape[0], X_hist_test.shape[1], 1)

    X_future_test_scaled = scaler_weather.transform(
        X_future_test.reshape(-1, X_future_test.shape[2])
    ).reshape(X_future_test.shape)

    y_test_scaled = scaler_water.transform(
        y_test.reshape(-1, 1)
    ).reshape(y_test.shape)

    print("Creating datasets...")
    train_dataset = WaterLevelDataset(X_hist_train_scaled, X_future_train_scaled, y_train_scaled)
    test_dataset = WaterLevelDataset(X_hist_test_scaled, X_future_test_scaled, y_test_scaled)
    print("Datasets created.")

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduced batch size
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)   # Reduced batch size
    print("Data loaders created.")

    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DeepARModel(
        weather_features=len(feature_names),
        hidden_size=32,  # Reduced from 64
        num_layers=2,
        dropout=0.3      # Increased dropout
    ).to(device)
    print("Model initialized.")

    print("Initializing optimizer and scheduler...")
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Lower LR, added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    print("Optimizer and scheduler initialized.")

    # Training loop with improved settings
    num_epochs = 30
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15  # Increased patience
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_predictions, val_targets = validate_epoch(model, test_loader, device)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  New best model saved at epoch {epoch+1}.")
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Patience Counter = {patience_counter}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
        
    # Plot actual vs predicted water levels (unscaled)
    print("Plotting predictions...")

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            historical = batch['historical'].to(device)
            future_weather = batch['future_weather'].to(device)
            target = batch['target'].to(device)

            mean, _ = model(historical, future_weather)
            all_preds.append(mean.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    preds = np.concatenate(all_preds).reshape(-1, 1)
    targets = np.concatenate(all_targets).reshape(-1, 1)

    # Inverse transform to original scale
    preds_unscaled = scaler_water.inverse_transform(preds)
    targets_unscaled = scaler_water.inverse_transform(targets)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(targets_unscaled, label='Actual', linewidth=2)
    plt.plot(preds_unscaled, label='Predicted', linewidth=2)
    plt.xlabel('Sample')
    plt.ylabel('Water Level')
    plt.title('Actual vs Predicted Water Levels')
    plt.legend()
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()

    print("Plot saved as 'actual_vs_predicted.png'.")

    print("Training completed!")

if __name__ == "__main__":
    main()