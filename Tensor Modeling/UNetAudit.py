import torch as th
import torch.nn as nn
import torch.nn.functional as F

th.set_printoptions(precision=3, sci_mode=False)

# ==========================================
# 1. 256x256 COMPATIBLE ARCHITECTURE
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        # --- STN Localization Layers ---
        self.loc_conv1 = nn.Conv2d(in_channels, 8, kernel_size=7, padding=3)
        self.loc_pool1 = nn.MaxPool2d(2, stride=2)
        
        self.loc_conv2 = nn.Conv2d(8, 10, kernel_size=5, padding=2)
        self.loc_pool2 = nn.MaxPool2d(2, stride=2)
        
        # --- STN Fully Connected Layers (Augmented for 256x256 Input) ---
        # 256x256 -> Pool1 -> 128x128 -> Pool2 -> 64x64
        # 10 channels * 64 * 64 = 40,960 input features
        self.loc_fc1 = nn.Linear(10 * 64 * 64, 32)
        self.loc_fc2 = nn.Linear(32, 2 * 3)
        
        # Initialize to Identity matrix
        self.loc_fc2.weight.data.zero_()
        self.loc_fc2.bias.data.copy_(th.tensor([1, 0, 0, 0, 1, 0], dtype=th.float32))

        # --- Standard U-Net Core Components ---
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t=None):
        print(f"--- Forward Pass Start ---")
        print(f"0. Input Tensor Shape:                      {list(x.shape)}")
        
        # ========================================================
        # INSIDE THE STN LOCALIZATION NETWORK
        # ========================================================
        print(f"\n--- [STN Internal Steps] ---")
        
        # Step 1: Conv1 (256x256 -> 256x256 due to padding=3)
        stn_features = F.relu(self.loc_conv1(x))
        print(f"STN Step 1 (loc_conv1 + ReLU):             {list(stn_features.shape)}")
        
        # Step 2: Pool1 (256x256 -> 128x128 due to stride=2)
        stn_features = self.loc_pool1(stn_features)
        print(f"STN Step 2 (loc_pool1):                    {list(stn_features.shape)}")
        
        # Step 3: Conv2 (128x128 -> 128x128 due to padding=2)
        stn_features = F.relu(self.loc_conv2(stn_features))
        print(f"STN Step 3 (loc_conv2 + ReLU):             {list(stn_features.shape)}")
        
        # Step 4: Pool2 (128x128 -> 64x64 due to stride=2)
        stn_features = self.loc_pool2(stn_features)
        print(f"STN Step 4 (loc_pool2):                    {list(stn_features.shape)}")
        
        # Step 5: Flattening/Flatten (Flattening to match 40,960)
        stn_flattened = stn_features.view(-1, 10 * 64 * 64)
        print(f"STN Step 5 (Flattened for Linear):          {list(stn_flattened.shape)}")
        
        # Step 6: First Linear Layer
        stn_fc1_out = F.relu(self.loc_fc1(stn_flattened))
        print(f"STN Step 6 (loc_fc1 + ReLU):                {list(stn_fc1_out.shape)}")
        
        # Step 7: Output Linear Layer
        theta_flat = self.loc_fc2(stn_fc1_out)
        print(f"STN Step 7 (loc_fc2 Raw Output):            {list(theta_flat.shape)}")
        
        # Step 8: Reshaping to a 2x3 Matrix
        theta = theta_flat.view(-1, 2, 3)
        print(f"STN Step 8 (Reshaped to Theta Matrix):\n{theta}")
        print(f"----------------------------\n")
        
        # ========================================================
        # AFFINE GRID GENERATION & BILINEAR SAMPLING
        # ========================================================
        grid = F.affine_grid(theta, size=x.size(), align_corners=False)
        print(f"1. Affine Grid Generated. Shape:            {list(grid.shape)}")
        print(f"   -> Center Target Lookup (X,Y):           ({grid[0, 128, 128, 0]:.3f}, {grid[0, 128, 128, 1]:.3f})")
        
        x_transformed = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        print(f"2. Post-Bilinear Sample Shape:              {list(x_transformed.shape)}")

        # ========================================================
        # STANDARD ENCODER & DECODER PATHS
        # ========================================================
        x1 = F.relu(self.enc1(x_transformed))
        print(f"3. After enc1 + ReLU (x1):                  {list(x1.shape)}")
        
        pooled = F.max_pool2d(x1, 2)
        print(f"4. After max_pool2d (downsample to 128):    {list(pooled.shape)}")
        
        x2 = F.relu(self.enc2(pooled))
        print(f"5. After enc2 + ReLU (x2):                  {list(x2.shape)}")
        
        u1 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        print(f"6. After interpolate (upsample to 256 u1):  {list(u1.shape)}")
        
        u1_processed = F.relu(self.dec2(u1))
        print(f"7. After dec2 + ReLU:                       {list(u1_processed.shape)}")
        
        x3 = u1_processed + x1
        print(f"8. After skip addition (x3):                {list(x3.shape)}")
        
        out = self.dec1(x3)
        print(f"9. Final Output Shape:                      {list(out.shape)}")
        print(f"--- Forward Pass End ---\n")
        
        return out

# ==========================================
# 2. VERIFICATION EXECUTION
# ==========================================
if __name__ == "__main__":
    # Updated dummy tensor: 1 Batch, 4 Channels, 256x256 Image Size
    dummy_input = th.randn(1, 4, 256, 256)
    
    model = SimpleUNet()
    output = model(dummy_input)
