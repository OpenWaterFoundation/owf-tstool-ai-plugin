import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class WaterLevelDataset(Dataset):
    """Custom dataset for water level forecasting"""
    def __init__(self, X_historical, X_future_weather, y):
        self.X_historical = torch.FloatTensor(X_historical)
        self.X_future_weather = torch.FloatTensor(X_future_weather)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X_historical)
    
    def __getitem__(self, idx):
        return {
            'historical': self.X_historical[idx],
            'future_weather': self.X_future_weather[idx],
            'target': self.y[idx]
        }

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for better feature interaction"""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        return torch.matmul(attention, V)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention = self.scaled_dot_product_attention(Q, K, V, mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(attention)

class ImprovedWaterLevelModel(nn.Module):
    """Improved water level forecasting model with attention and residual connections"""
    def __init__(self, weather_features, hidden_size=128, num_layers=3, dropout=0.4):
        super().__init__()
        
        self.weather_features = weather_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection layers
        self.historical_projection = nn.Linear(weather_features + 1, hidden_size)
        self.weather_projection = nn.Linear(weather_features, hidden_size)
        
        # Historical encoder with bidirectional LSTM
        self.historical_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads=8)
        
        # Context processing
        self.context_norm = nn.LayerNorm(hidden_size * 2)
        self.context_ff = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2)
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_size + hidden_size * 2,  # weather + context
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers with skip connections
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using He initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif param.dim() >= 2:
                    nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, historical, future_weather):
        batch_size = historical.size(0)
        
        # Project inputs
        historical_proj = self.historical_projection(historical)
        weather_proj = self.weather_projection(future_weather)
        
        # Encode historical data
        historical_encoded, (hidden, cell) = self.historical_encoder(historical_proj)
        
        # Apply attention to historical features
        attended_hist = self.attention(historical_encoded, historical_encoded, historical_encoded)
        
        # Residual connection and normalization
        attended_hist = self.context_norm(attended_hist + historical_encoded)
        
        # Feed forward
        ff_output = self.context_ff(attended_hist)
        context_features = self.context_norm(ff_output + attended_hist)
        
        # Use last context as initial context for decoder
        context = context_features[:, -1:, :].repeat(1, future_weather.size(1), 1)
        
        # Combine context with future weather
        decoder_input = torch.cat([weather_proj, context], dim=2)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input)
        
        # Normalize and generate output
        decoder_output = self.output_norm(decoder_output)
        output = self.output_layers(decoder_output)
        
        return output.squeeze(-1)

def create_model_with_proper_loss():
    """Create model with appropriate loss function"""
    return nn.MSELoss()

def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch in dataloader:
        historical = batch['historical'].to(device)
        future_weather = batch['future_weather'].to(device)
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(historical, future_weather)
        loss = criterion(predictions, target)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        if batch_count % 10 == 0:
            print(f"  [train_epoch] Batch {batch_count}: Loss = {loss.item():.6f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"  [train_epoch] Epoch finished. Avg Loss = {avg_loss:.6f}")
    return avg_loss

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            historical = batch['historical'].to(device)
            future_weather = batch['future_weather'].to(device)
            target = batch['target'].to(device)
            
            pred = model(historical, future_weather)
            loss = criterion(pred, target)
            
            total_loss += loss.item()
            predictions.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    print(f"  [validate_epoch] Epoch finished. Avg Loss = {avg_loss:.6f}")
    return avg_loss, predictions, targets

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

def main():
    print("=== Advanced Water Level Forecasting Model ===")
    
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
    
    # Improved data preprocessing
    print("Preprocessing data...")
    
    # Use MinMaxScaler for better stability
    scaler_weather = StandardScaler()
    scaler_water = StandardScaler()
    
    # Fit scalers on training data
    weather_data_train = X_hist_train[:, :, :-1].reshape(-1, len(feature_names))
    water_data_train = X_hist_train[:, :, -1].reshape(-1, 1)
    
    scaler_weather.fit(weather_data_train)
    scaler_water.fit(water_data_train)
    
    # Transform training data
    X_hist_train_scaled = X_hist_train.copy()
    X_hist_train_scaled[:, :, :-1] = scaler_weather.transform(
        X_hist_train[:, :, :-1].reshape(-1, len(feature_names))
    ).reshape(X_hist_train.shape[0], X_hist_train.shape[1], -1)
    X_hist_train_scaled[:, :, -1] = scaler_water.transform(
        X_hist_train[:, :, -1].reshape(-1, 1)
    ).reshape(X_hist_train.shape[0], X_hist_train.shape[1])
    
    X_future_train_scaled = scaler_weather.transform(
        X_future_train.reshape(-1, len(feature_names))
    ).reshape(X_future_train.shape)
    
    y_train_scaled = scaler_water.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    
    # Transform test data
    X_hist_test_scaled = X_hist_test.copy()
    X_hist_test_scaled[:, :, :-1] = scaler_weather.transform(
        X_hist_test[:, :, :-1].reshape(-1, len(feature_names))
    ).reshape(X_hist_test.shape[0], X_hist_test.shape[1], -1)
    X_hist_test_scaled[:, :, -1] = scaler_water.transform(
        X_hist_test[:, :, -1].reshape(-1, 1)
    ).reshape(X_hist_test.shape[0], X_hist_test.shape[1])
    
    X_future_test_scaled = scaler_weather.transform(
        X_future_test.reshape(-1, len(feature_names))
    ).reshape(X_future_test.shape)
    
    y_test_scaled = scaler_water.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = WaterLevelDataset(X_hist_train_scaled, X_future_train_scaled, y_train_scaled)
    test_dataset = WaterLevelDataset(X_hist_test_scaled, X_future_test_scaled, y_test_scaled)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ImprovedWaterLevelModel(
        weather_features=len(feature_names),
        hidden_size=128,
        num_layers=3,
        dropout=0.2
    ).to(device)
    
    # Initialize optimizer and loss
    criterion = create_model_with_proper_loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Training parameters
    num_epochs = 60
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_metrics = None
    patience = 5
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_predictions, val_targets = validate_epoch(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Calculate metrics on original scale
        val_pred_unscaled = scaler_water.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
        val_target_unscaled = scaler_water.inverse_transform(val_targets.reshape(-1, 1)).flatten()
        
        metrics = calculate_metrics(val_target_unscaled, val_pred_unscaled)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'metrics': metrics
            }, 'best_water_level_model.pth')
            
            # Save scalers
            joblib.dump(scaler_weather, 'scaler_weather.pkl')
            joblib.dump(scaler_water, 'scaler_water.pkl')
            
            print(f"  New best model saved! Metrics: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")
        else:
            patience_counter += 1
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, LR={current_lr:.6f}")
        print(f"  Metrics: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}, MAPE={metrics['MAPE']:.2f}%")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load('best_water_level_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler_weather = joblib.load('scaler_weather.pkl')
    scaler_water = joblib.load('scaler_water.pkl')
    
    # Generate final predictions
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            historical = batch['historical'].to(device)
            future_weather = batch['future_weather'].to(device)
            target = batch['target'].to(device)
            
            pred = model(historical, future_weather)
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Transform back to original scale
    predictions_unscaled = scaler_water.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets_unscaled = scaler_water.inverse_transform(targets.reshape(-1, 1)).flatten()
    
    # Calculate final metrics
    final_metrics = calculate_metrics(targets_unscaled, predictions_unscaled)
    
    print(f"\n=== Final Test Results ===")
    print(f"RMSE: {final_metrics['RMSE']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.4f}")
    print(f"R²: {final_metrics['R2']:.4f}")
    print(f"MAPE: {final_metrics['MAPE']:.2f}%")
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Training Loss', alpha=0.8)
    axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Predictions vs actual (scatter plot)
    axes[0, 1].scatter(targets_unscaled, predictions_unscaled, alpha=0.6, s=1)
    axes[0, 1].plot([targets_unscaled.min(), targets_unscaled.max()], 
                    [targets_unscaled.min(), targets_unscaled.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Water Level')
    axes[0, 1].set_ylabel('Predicted Water Level')
    axes[0, 1].set_title(f'Predictions vs Actual (R²={final_metrics["R2"]:.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Time series plot (sample)
    sample_size = min(1000, len(targets_unscaled))
    sample_idx = np.random.choice(len(targets_unscaled), sample_size, replace=False)
    sample_idx = np.sort(sample_idx)
    
    axes[1, 0].plot(sample_idx, targets_unscaled[sample_idx], label='Actual', alpha=0.8, linewidth=1)
    axes[1, 0].plot(sample_idx, predictions_unscaled[sample_idx], label='Predicted', alpha=0.8, linewidth=1)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Water Level')
    axes[1, 0].set_title('Time Series Comparison (Sample)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = targets_unscaled - predictions_unscaled
    axes[1, 1].scatter(predictions_unscaled, residuals, alpha=0.6, s=1)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Water Level')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title(f'Residuals Plot (MAE={final_metrics["MAE"]:.4f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_water_level_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nResults saved as 'advanced_water_level_results.png'")
    print("Training completed successfully!")

if __name__ == "__main__":
    main()