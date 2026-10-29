# -*- coding: utf-8 -*-

import arcpy
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve shared model helpers beside this toolbox, including when ArcGIS loads by path.
import importlib.util as _importlib_util
from pathlib import Path as _ModelPath
_model_spec = _importlib_util.spec_from_file_location(
    "_chm_models", _ModelPath(__file__).resolve().with_name("chm_models.py"))
_chm_models = _importlib_util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_chm_models)

INPUT_NORMALIZATION_DIVISOR = 255.0

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

def publish_geodatabase_raster(tiff_path, output, overwrite):
    """Convert to a staged FGDB raster, then publish with rollback on failure."""
    workspace = os.path.dirname(output)
    staging = arcpy.CreateUniqueName("chm_pending", workspace)
    backup = None
    try:
        with arcpy.EnvManager(overwriteOutput=False, extent=None, cellSize=None,
                              outputCoordinateSystem=None, snapRaster=None):
            arcpy.management.CopyRaster(tiff_path, staging, pixel_type="32_BIT_FLOAT",
                                        nodata_value="-9999", scale_pixel_value="NONE")
            if arcpy.Exists(output):
                if not overwrite:
                    raise FileExistsError("Output exists and overwrite is disabled.")
                backup = arcpy.CreateUniqueName("chm_previous", workspace)
                arcpy.management.Rename(output, backup, "RasterDataset")
            try:
                arcpy.management.Rename(staging, output, "RasterDataset")
                staging = None
            except Exception:
                if backup and not arcpy.Exists(output):
                    try:
                        arcpy.management.Rename(backup, output, "RasterDataset")
                        backup = None
                    except Exception as exc:
                        arcpy.AddError(f"Could not restore the previous output. It remains at {backup}: {exc}")
                raise
        if backup:
            try:
                arcpy.management.Delete(backup)
                backup = None
            except Exception as exc:
                arcpy.AddWarning(f"New output saved; previous raster retained at {backup}: {exc}")
    finally:
        if staging and arcpy.Exists(staging):
            try:
                arcpy.management.Delete(staging)
            except Exception as exc:
                arcpy.AddWarning(f"Could not remove temporary geodatabase raster {staging}: {exc}")


class PredictCHM:
    def __init__(self):
        self.label = "Predict CHM from Imagery"
        self.description = "Uses a trained UNet, ResUNet, or UNet++ model to generate a Canopy Height Model from a 4-band raster."
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
        model_type.filter.list = list(_chm_models.MODEL_TYPES)
        model_type.value = "UNet"
        model_type.enabled = True
        
        normalization_factor = arcpy.Parameter(
            displayName="Input Normalization Divisor (Fixed at 255)",
            name="normalization_factor",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        normalization_factor.value = INPUT_NORMALIZATION_DIVISOR
        normalization_factor.enabled = False

        out_raster = arcpy.Parameter(
            displayName="Output Predicted CHM (TIFF or File Geodatabase Raster)",
            name="out_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )

        return [in_raster, model_weights, model_type, normalization_factor, out_raster]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        parameters[3].value = INPUT_NORMALIZATION_DIVISOR
        parameters[3].enabled = False

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        try:
            self._execute(parameters, messages)
        finally:
            arcpy.ResetProgressor()

    def _execute(self, parameters, messages):
        raw_in_raster = parameters[0].valueAsText
        model_weights = parameters[1].valueAsText
        model_choice = parameters[2].valueAsText
        norm_factor = INPUT_NORMALIZATION_DIVISOR
        raw_out_raster = parameters[4].valueAsText
        in_raster = arcpy.Describe(raw_in_raster).catalogPath
        if not raw_out_raster:
            raise ValueError("Choose an output TIFF or a raster name inside an existing file geodatabase.")
        out_raster = os.path.abspath(raw_out_raster)
        output_folder, output_name = os.path.split(out_raster)
        is_geodatabase = output_folder.lower().endswith(".gdb")
        if is_geodatabase:
            if not arcpy.Exists(output_folder):
                raise ValueError("Choose an existing file geodatabase.")
            if not output_name or arcpy.ValidateTableName(output_name, output_folder) != output_name or "." in output_name:
                raise ValueError("Use a valid geodatabase raster name without a file extension, for example project.gdb/predicted_chm.")
            temporary_folder = arcpy.env.scratchFolder or tempfile.gettempdir()
        else:
            if any(part.lower().endswith(".gdb") for part in Path(out_raster).parts):
                raise ValueError("Place the raster directly inside the file geodatabase and provide its dataset name.")
            if not out_raster.lower().endswith((".tif", ".tiff")):
                out_raster += ".tif"
            output_folder = os.path.dirname(out_raster)
            if not os.path.isdir(output_folder):
                raise ValueError("Choose an existing output folder for the TIFF.")
            if os.path.isdir(out_raster):
                raise ValueError("Output must be a raster filename, not a directory.")
            temporary_folder = output_folder
        for protected in (in_raster, model_weights):
            same = os.path.normcase(os.path.realpath(out_raster)) == os.path.normcase(os.path.realpath(protected))
            if not same and os.path.exists(out_raster) and os.path.exists(protected):
                same = os.path.samefile(out_raster, protected)
            if same:
                raise ValueError("Output must be different from the source imagery and model weights.")
        overwrite = bool(arcpy.env.overwriteOutput)
        if (arcpy.Exists(out_raster) if is_geodatabase else os.path.exists(out_raster)) and not overwrite:
            raise FileExistsError("Output exists. Choose a new name or enable ArcGIS overwrite output.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arcpy.AddMessage(f"Initializing inference on device: {device}")
        arcpy.AddMessage(f"Loading {model_choice} architecture and trained weights...")
        model = _chm_models.build_chm_model(model_choice, UNetRegressor).to(device)
        _chm_models.load_chm_weights(model, model_weights, model_choice)
        model.eval()
        arcpy.AddMessage(f"Opening input raster: {in_raster}")
        temporary = None
        try:
            with rasterio.open(in_raster) as src:
                if src.count < 4:
                    raise ValueError(f"Input raster must have at least 4 bands. Found {src.count} bands.")

                # Copy spatial metadata for the output
                meta = src.meta.copy()
                meta.update({
                    "count": 1, 
                    "dtype": rasterio.float32, 
                    "nodata": -9999.0,
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
                with tempfile.NamedTemporaryFile(dir=temporary_folder, prefix=".chm_", suffix=".tif", delete=False) as handle:
                    temporary = handle.name
                with rasterio.open(temporary, 'w', **meta) as dst:
                    with torch.no_grad():
                        # Slide the window using the smaller step size
                        for y in range(0, height, step):
                            for x in range(0, width, step):
                                window = Window(x, y, patch_size, patch_size)
                                
                                # Read patch and normalize (src is still open here!)
                                masked_patch = src.read(indexes=[1, 2, 3, 4], window=window,
                                                        boundless=True, masked=True).astype(np.float32)
                                patch = np.asarray(masked_patch.filled(0), dtype=np.float32)
                                # Match training: all-four-band zero pixels are padding.
                                valid_patch = ((~np.ma.getmaskarray(masked_patch) & np.isfinite(patch)).all(axis=0)
                                               & np.any(patch != 0, axis=0))
                                patch[:, ~valid_patch] = 0.0
                                patch = patch / norm_factor

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
                                    valid_output = valid_patch[crop_y1:crop_y2, crop_x1:crop_x2]
                                    out_patch = np.where(valid_output[None] & np.isfinite(out_patch),
                                                         out_patch, -9999.0).astype(np.float32)
                                    out_window = Window(write_x, write_y, write_w, write_h)
                                    dst.write(out_patch, window=out_window)

                                # Update Progressor
                                patch_count += 1
                                if patch_count % 10 == 0 or patch_count == total_patches:
                                    arcpy.SetProgressorPosition(patch_count)
            # Close readers/writers before publishing to either destination.
            if is_geodatabase:
                arcpy.SetProgressor("default", "Saving predicted CHM to the geodatabase...")
                publish_geodatabase_raster(temporary, out_raster, overwrite)
            else:
                if overwrite:
                    os.replace(temporary, out_raster)
                else:
                    os.rename(temporary, out_raster)
                temporary = None
                for suffix in (".aux.xml", ".ovr", ".msk"):
                    sidecar = out_raster + suffix
                    if os.path.isfile(sidecar):
                        try:
                            os.remove(sidecar)
                        except OSError as exc:
                            arcpy.AddWarning(f"Could not remove outdated raster sidecar {sidecar}: {exc}")
        except Exception as exc:
            arcpy.AddError(f"CHM prediction failed: {exc}")
            raise
        finally:
            if temporary:
                for path in (temporary, temporary + ".aux.xml", temporary + ".ovr", temporary + ".msk"):
                    if os.path.isfile(path):
                        try:
                            os.remove(path)
                        except OSError as exc:
                            arcpy.AddWarning(f"Could not remove temporary file {path}: {exc}")
        parameters[4].value = out_raster
        arcpy.AddMessage(f"Inference complete! Output saved to: {out_raster}")
        return

    def postExecute(self, parameters):
        return
