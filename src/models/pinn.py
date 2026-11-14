import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# NEW: Fourier Feature Mapping to solve Spectral Bias (learning sharp peaks)
class FourierFeatures(nn.Module):
    def __init__(self, num_frequencies=4):
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, x):
        # Create a range of frequencies
        freqs = 2.0 ** torch.arange(self.num_frequencies, device=x.device)
        x_ff = x * freqs * np.pi
        return torch.cat([x, torch.sin(x_ff), torch.cos(x_ff)], dim=1)

class FlamePINN(nn.Module):
    def __init__(self):
        super(FlamePINN, self).__init__()
        self.fourier = FourierFeatures(num_frequencies=4) # Outputs 9 features (1 + 4 sin + 4 cos)
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, t):
        x = self.fourier(t)
        return self.net(x)

def train_pinn(df, epochs=5000):
    t_data = df['frame_idx'].values.reshape(-1, 1).astype(np.float32)
    t_data = (t_data - t_data.min()) / (t_data.max() - t_data.min()) 
    
    I_raw = df['intensity_3'].values.reshape(-1, 1).astype(np.float32)
    I_max = np.max(I_raw)
    I_data = I_raw / I_max
    
    t_tensor = torch.from_numpy(t_data) 
    I_tensor = torch.from_numpy(I_data)

    model = FlamePINN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        I_pred = model(t_tensor)
        
        # 1. Data Loss (Standard MSE)
        data_loss = nn.MSELoss()(I_pred, I_tensor)
        
        # 2. Physics Loss: Bounded Energy & Non-negativity
        # Rule A: Flame intensity cannot be negative (relu(-I_pred))
        # Rule B: Flame intensity cannot exceed the theoretical maximum (I_pred - 1.0)
        loss_neg = torch.mean(torch.relu(-I_pred))
        loss_upper = torch.mean(torch.relu(I_pred - 1.0))
        
        physics_loss = loss_neg + loss_upper
        
        # Combined Loss
        total_loss = data_loss + 0.5 * physics_loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Data Loss {data_loss.item():.6f}, Physics Bounds Loss {physics_loss.item():.6f}")
            
    return model, t_tensor.detach().numpy()

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.data.loader import load_slice_timeseries
    
    df = load_slice_timeseries("data/raw/SLICE_GMT079/pic_info_059_noBG.xls")
    df = df.dropna(subset=['intensity_3']) 
    print("Training PINN (Fourier Features)...")
    model, t_norm = train_pinn(df)
    torch.save(model.state_dict(), "src/models/pinn_weights.pth")
    print("Model saved!")