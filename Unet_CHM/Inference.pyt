# -*- coding: utf-8 -*-

import arcpy
import os
import sys
import numpy as np
import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 1. Model Architectures (Must match training definitions)
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
# 2. ArcGIS Python Toolbox Definitions
# =========================================================
class Toolbox:
    def __init__(self):
        self.label = "CHM Inference Toolbox"
        self.alias = "chm_inference"
        self.tools = [PredictCHM]

class PredictCHM:
    def __init__(self):
        self.label = "Predict CHM from Imagery"
        self.description = "Uses a trained UNet model to generate a Canopy Height Model from a 4-band raster."
        self.canRunInBackground = False

    def getParameterInfo(self):
        in_raster = arcpy.Parameter(
            displayName="Input 4-Band Imagery (.tif)",
            name="in_raster",
            datatype=["DERasterDataset", "GPRasterLayer"], # Supports layers loaded in map and direct datasets
            parameterType="Required",
            direction="Input"
        )
        
        model_weights = arcpy.Parameter(
            displayName="Trained Model Weights (.pth)",
            name="model_weights",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )

        model_type = arcpy.Parameter(
            displayName="Model Architecture",
            name="model_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        model_type.filter.list = ["UNet"]
        model_type.value = "UNet"
        model_type.enabled = False  # Preserve existing parameter positions.
        
        normalization_factor = arcpy.Parameter(
            displayName="Input Normalization Divisor (e.g., 255 or 65535)",
            name="normalization_factor",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        normalization_factor.value = 255.0

        out_raster = arcpy.Parameter(
            displayName="Output Predicted CHM (.tif)",
            name="out_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )

        return [in_raster, model_weights, model_type, normalization_factor, out_raster]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        # 1. Resolve parameters
        raw_in_raster = parameters[0].valueAsText
        model_weights = parameters[1].valueAsText
        model_choice = parameters[2].valueAsText
        norm_factor = parameters[3].value
        raw_out_raster = parameters[4].valueAsText

        # Extract strict file paths for rasterio (handles Catalog Pane dragging & Map layers)
        try:
            in_raster = arcpy.Describe(raw_in_raster).catalogPath
        except Exception as e:
            arcpy.AddError(f"Could not resolve the absolute path for the input raster. Ensure it is a valid file. Error: {e}")
            return
            
        out_raster = raw_out_raster
        if not out_raster.lower().endswith(".tif"):
            out_raster += ".tif"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arcpy.AddMessage(f"Initializing inference on device: {device}")

        # 2. Initialize and Load Model
        arcpy.AddMessage("Loading model architecture and trained weights...")
        try:
            if model_choice == "UNet":
                model = UNetRegressor(in_channels=4, out_channels=1).to(device)
            else:
                raise ValueError("Only UNet is supported by the CHM tools.")

            model.load_state_dict(torch.load(model_weights, map_location=device))
            model.eval() # CRITICAL: Set to evaluation mode
        except Exception as e:
            arcpy.AddError(f"Failed to load model weights. Ensure architecture matches the .pth file. Error: {e}")
            return

        patch_size = 256

        # 3. Open input raster and prepare output raster

        # 3. Open input raster and prepare output raster
        arcpy.AddMessage(f"Opening input raster: {in_raster}")
        try:
            # ---> SOURCE BLOCK STARTS HERE <---
            with rasterio.open(in_raster) as src:
                if src.count < 4:
                    arcpy.AddError(f"Input raster must have at least 4 bands. Found {src.count} bands.")
                    return

                # Copy spatial metadata for the output
                meta = src.meta.copy()
                meta.update({
                    "count": 1, 
                    "dtype": rasterio.float32, 
                    "nodata": 0.0,
                    "compress": "lzw" # Compress output to save space
                })

                width = src.width
                height = src.height
                
                # Overlap variables
                patch_size = 256
                margin = 32
                step = patch_size - (2 * margin) 

                total_patches = int(np.ceil(width / step) * np.ceil(height / step))
                arcpy.SetProgressor("step", "Predicting CHM Patches (with Overlap)...", 0, total_patches, 1)

                patch_count = 0
                
                # ---> DESTINATION BLOCK MUST BE INDENTED INSIDE THE SOURCE BLOCK <---
                with rasterio.open(out_raster, 'w', **meta) as dst:
                    with torch.no_grad():
                        # Slide the window using the smaller step size
                        for y in range(0, height, step):
                            for x in range(0, width, step):
                                window = Window(x, y, patch_size, patch_size)
                                
                                # Read patch and normalize (src is still open here!)
                                patch = src.read(window=window, boundless=True, fill_value=0).astype(np.float32)
                                patch = patch[:4, :, :] 
                                patch = patch / norm_factor
                                patch = np.nan_to_num(patch, nan=0.0)

                                # Predict
                                patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device) 
                                pred_tensor = model(patch_tensor) 
                                pred_np = pred_tensor.squeeze(0).cpu().numpy() # [1, 256, 256]

                                # --- Cropping Logic to Remove Edge Artifacts ---
                                crop_y1 = 0 if y == 0 else margin
                                crop_x1 = 0 if x == 0 else margin
                                
                                write_y = y + crop_y1
                                write_x = x + crop_x1
                                
                                write_h = min(patch_size - crop_y1 - (margin if (y + patch_size) < height else 0), height - write_y)
                                write_w = min(patch_size - crop_x1 - (margin if (x + patch_size) < width else 0), width - write_x)

                                if write_h > 0 and write_w > 0:
                                    crop_y2 = crop_y1 + write_h
                                    crop_x2 = crop_x1 + write_w
                                    
                                    out_patch = pred_np[:, crop_y1:crop_y2, crop_x1:crop_x2]
                                    out_window = Window(write_x, write_y, write_w, write_h)
                                    dst.write(out_patch, window=out_window)

                                # Update Progressor
                                patch_count += 1
                                if patch_count % 10 == 0:
                                    arcpy.SetProgressorPosition(patch_count)
        except Exception as e:
            arcpy.AddError(f"Raster processing failed: {e}")
        #     return
        # arcpy.AddMessage(f"Opening input raster: {in_raster}")
        # try:
        #     with rasterio.open(in_raster) as src:
        #         if src.count < 4:
        #             arcpy.AddError(f"Input raster must have at least 4 bands. Found {src.count} bands.")
        #             return

        #         # Copy spatial metadata for the output
        #         meta = src.meta.copy()
        #         meta.update({
        #             "count": 1, 
        #             "dtype": rasterio.float32, 
        #             "nodata": 0.0,
        #             "compress": "lzw" # Compress output to save space
        #         })

        #         width = src.width
        #         height = src.height
        #         total_patches = int(np.ceil(width / patch_size) * np.ceil(height / patch_size))
        #         arcpy.SetProgressor("step", "Predicting CHM Patches...", 0, total_patches, 1)

        #         # 4. Patch-based Inference Loop
        #         # 3. Patch-based Overlapping Inference Loop
        #     patch_size = 256
        #     margin = 32
        #     step = patch_size - (2 * margin) # Move by 192 pixels to create overlap

        #     total_patches = int(np.ceil(width / step) * np.ceil(height / step))
        #     arcpy.SetProgressor("step", "Predicting CHM Patches (with Overlap)...", 0, total_patches, 1)

        #     patch_count = 0
        #     with rasterio.open(out_raster, 'w', **meta) as dst:
        #         with torch.no_grad():
        #             # Slide the window using the smaller step size
        #             for y in range(0, height, step):
        #                 for x in range(0, width, step):
        #                     window = Window(x, y, patch_size, patch_size)
                            
        #                     # Read patch and normalize
        #                     patch = src.read(window=window, boundless=True, fill_value=0).astype(np.float32)
        #                     patch = patch[:4, :, :] 
        #                     patch = patch / norm_factor
        #                     patch = np.nan_to_num(patch, nan=0.0)

        #                     # Predict
        #                     patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device) 
        #                     pred_tensor = model(patch_tensor) 
        #                     pred_np = pred_tensor.squeeze(0).cpu().numpy() # [1, 256, 256]

        #                     # --- Cropping Logic to Remove Edge Artifacts ---
        #                     # Keep the outer margin if we are at the absolute top/left edge of the image
        #                     crop_y1 = 0 if y == 0 else margin
        #                     crop_x1 = 0 if x == 0 else margin
                            
        #                     write_y = y + crop_y1
        #                     write_x = x + crop_x1
                            
        #                     # Calculate valid write dimensions (discarding the far margin unless at the bottom/right edge)
        #                     write_h = min(patch_size - crop_y1 - (margin if (y + patch_size) < height else 0), height - write_y)
        #                     write_w = min(patch_size - crop_x1 - (margin if (x + patch_size) < width else 0), width - write_x)

        #                     if write_h > 0 and write_w > 0:
        #                         crop_y2 = crop_y1 + write_h
        #                         crop_x2 = crop_x1 + write_w
                                
        #                         out_patch = pred_np[:, crop_y1:crop_y2, crop_x1:crop_x2]
        #                         out_window = Window(write_x, write_y, write_w, write_h)
        #                         dst.write(out_patch, window=out_window)

        #                     # Update Progressor
        #                     patch_count += 1
        #                     if patch_count % 10 == 0:
        #                         arcpy.SetProgressorPosition(patch_count)
        # except Exception as e:
        #     arcpy.AddError(f"Raster processing failed: {e}")
        #     return

        arcpy.AddMessage(f"Inference complete! Output saved to: {out_raster}")
        return

    def postExecute(self, parameters):
        return