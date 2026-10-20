# -*- coding: utf-8 -*-

import arcpy
import os
import math
import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================================================
# 1. Custom SSIM & Combined Loss
# =========================================================
def gaussian_window(window_size, sigma):
    """Generates a 1D Gaussian kernel."""
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    """Creates a 2D Gaussian window for Conv2d."""
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, data_range=1.0, valid_mask=None):
    """SSIM using only valid pixels in each local Gaussian window."""
    mask = torch.ones_like(img2, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    if not mask.any():
        raise ValueError("No valid CHM pixels in this batch.")
    img1 = torch.where(mask, img1, torch.zeros_like(img1))
    img2 = torch.where(mask, img2, torch.zeros_like(img2))
    weights = mask.to(img1.dtype)
    channel = img1.size(1)
    window = create_window(window_size, channel).to(device=img1.device, dtype=img1.dtype)
    def local_sum(value):
        return F.conv2d(value, window, padding=window_size//2, groups=channel)
    denominator = local_sum(weights).clamp_min(1e-12)
    mu1 = local_sum(img1) / denominator
    mu2 = local_sum(img2) / denominator
    variance1 = (local_sum(img1.square()) / denominator - mu1.square()).clamp_min(0)
    variance2 = (local_sum(img2.square()) / denominator - mu2.square()).clamp_min(0)
    covariance = local_sum(img1 * img2) / denominator - mu1 * mu2
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mu1 * mu2 + c1) * (2 * covariance + c2)) / (
        (mu1.square() + mu2.square() + c1) * (variance1 + variance2 + c2))
    return score[mask].mean()


class CombinedCHMLoss(nn.Module):
    def __init__(self, alpha=0.85, data_range=40.0, window_size=11):
        super().__init__()
        self.alpha = alpha
        self.data_range = data_range
        self.window_size = window_size

    def forward(self, y_pred, y_true, valid_mask=None):
        mask = torch.isfinite(y_true) if valid_mask is None else valid_mask.bool() & torch.isfinite(y_true)
        if not mask.any():
            raise ValueError("No valid CHM pixels in this batch.")
        if not torch.isfinite(y_pred[mask]).all():
            raise ValueError("Model produced nonfinite heights at valid pixels.")
        l1_loss = F.smooth_l1_loss(y_pred[mask], y_true[mask])
        ssim_value = ssim(y_pred, y_true, self.window_size, self.data_range, mask)
        return self.alpha * l1_loss + (1.0 - self.alpha) * (1.0 - ssim_value)

# =========================================================
# 2. PyTorch Dataset 
# =========================================================
class CHMDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        img_files = set(f for f in os.listdir(image_dir) if f.endswith('.tif'))
        lbl_files = set(f for f in os.listdir(label_dir) if f.endswith('.tif'))
        self.filenames = list(img_files.intersection(lbl_files))
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, file_name)
        lbl_path = os.path.join(self.label_dir, file_name)

        with rasterio.open(img_path) as src:
            masked_image = src.read(masked=True).astype(np.float32)
        image = np.asarray(masked_image.filled(0), dtype=np.float32)
        # ArcGIS-exported padding has ordinary zeros in every imagery band.
        # Do not infer missing coverage from zero CHM heights.
        image_valid = ((~np.ma.getmaskarray(masked_image) & np.isfinite(image)).all(axis=0)
                       & np.any(image != 0, axis=0))
        image[:, ~image_valid] = 0.0
        image /= 255.0

        with rasterio.open(lbl_path) as src:
            masked_label = src.read(1, masked=True).astype(np.float32)
        label = np.asarray(masked_label.filled(0), dtype=np.float32)
        valid = image_valid & ~np.ma.getmaskarray(masked_label) & np.isfinite(label)
        label = np.where(valid, np.maximum(label, 0.0), 0.0).astype(np.float32)
        return (torch.from_numpy(image), torch.from_numpy(label[None]),
                torch.from_numpy(valid[None]))

# =========================================================
# 3. Model Architectures
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNetRegressor(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024)) 

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return self.relu(logits)


# =========================================================
# 4. ArcGIS Python Toolbox Definitions
# =========================================================
class Toolbox:
    def __init__(self):
        self.label = "CHM Prediction Toolbox"
        self.alias = "chm_tools"
        self.tools = [TrainCHM]

class TrainCHM:
    def __init__(self):
        self.label = "Train CHM Regressor"
        self.description = "Trains a UNet model to predict Canopy Height Models from 4-band imagery."
        self.canRunInBackground = False

    def getParameterInfo(self):
        # 1. Inputs
        img_dir = arcpy.Parameter(
            displayName="Input Images Directory (4-Band .tif)",
            name="img_dir",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )
        
        lbl_dir = arcpy.Parameter(
            displayName="Input Labels Directory (1-Band CHM .tif)",
            name="lbl_dir",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        # 2. Model Settings
        model_type = arcpy.Parameter(
            displayName="Model Architecture",
            name="model_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        model_type.filter.list = ["UNet"]
        model_type.value = "UNet"
        # model_type.enabled = False  # Preserve existing parameter positions.

        # 3. Hyperparameters
        epochs = arcpy.Parameter(
            displayName="Number of Epochs",
            name="epochs",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        epochs.value = 50

        batch_size = arcpy.Parameter(
            displayName="Batch Size",
            name="batch_size",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        batch_size.value = 16

        # 4. Output File
        out_pth = arcpy.Parameter(
            displayName="Output Model Weights (.pth)",
            name="out_pth",
            datatype="DEFile",
            parameterType="Required",
            direction="Output"
        )

        return [img_dir, lbl_dir, model_type, epochs, batch_size, out_pth]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        image_dir = parameters[0].valueAsText
        label_dir = parameters[1].valueAsText
        model_choice = parameters[2].valueAsText
        num_epochs = parameters[3].value
        batch_size = parameters[4].value
        save_path = parameters[5].valueAsText

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arcpy.AddMessage(f"Starting training on device: {device}")

        # Instantiate Dataset
        full_dataset = CHMDataset(image_dir=image_dir, label_dir=label_dir)
        arcpy.AddMessage(f"Total valid tile pairs found: {len(full_dataset)}")

        if len(full_dataset) == 0:
            arcpy.AddError("No matching .tif files found between image and label directories.")
            return

        # Split Data
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        # DataLoaders (num_workers=0 to prevent Windows multiprocessing crash)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Initialize Model
        if model_choice == "UNet":
            model = UNetRegressor(in_channels=4, out_channels=1).to(device)
        else:
            raise ValueError("Only UNet is supported by the CHM tools.")

        # Initialize Loss, Optimizer, and Scheduler
        criterion = CombinedCHMLoss(alpha=0.85, data_range=40.0).to(device) 
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_loss = float('inf')

        # Training Loop
        for epoch in range(num_epochs):
            model.train() 
            train_loss = 0.0
            train_valid_pixels = 0
            
            for batch_idx, (images, labels, valid_mask) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                valid_mask = valid_mask.to(device)
                valid_count = int(valid_mask.sum().item())
                if not valid_count:
                    continue
                
                optimizer.zero_grad()
                predictions = model(images)
                loss = criterion(predictions, labels, valid_mask)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * valid_count
                train_valid_pixels += valid_count
                
            if not train_valid_pixels:
                raise ValueError("Training partition contains no valid CHM pixels.")
            avg_train_loss = train_loss / train_valid_pixels
            
            model.eval() 
            val_loss = 0.0
            val_valid_pixels = 0
            
            with torch.no_grad(): 
                for images, labels, valid_mask in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    valid_mask = valid_mask.to(device)
                    valid_count = int(valid_mask.sum().item())
                    if not valid_count:
                        continue
                    predictions = model(images)
                    loss = criterion(predictions, labels, valid_mask)
                    val_loss += loss.item() * valid_count
                    val_valid_pixels += valid_count
                    
            if not val_valid_pixels:
                raise ValueError("Validation partition contains no valid CHM pixels.")
            avg_val_loss = val_loss / val_valid_pixels
            scheduler.step(avg_val_loss)
            
            # Logging
            log_msg = f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            arcpy.AddMessage(log_msg)
            
            # Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                arcpy.AddMessage(f"  --> Best model saved to {save_path} (Val Loss improved to {best_val_loss:.4f})")

        arcpy.AddMessage("Training complete!")
        return

    def postExecute(self, parameters):
        return