# -*- coding: utf-8 -*-

import arcpy
import os
import math
import json
import csv
import time
import tempfile
from datetime import datetime, timezone
import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

INPUT_NORMALIZATION_DIVISOR = 255.0


def save_chm_metadata(weights_path, epoch, validation_loss, batch_size, pretrained_model=None, early_stopping_patience=5):
    """Record the fixed preprocessing beside each best U-Net checkpoint."""
    metadata = {
        "schema_version": 1,
        "task": "chm_regression",
        "model_type": "UNet",
        "weights_file": os.path.basename(weights_path),
        "input_channels": 4,
        "output_channels": 1,
        "preprocessing": {
            "band_indices": [1, 2, 3, 4],
            "normalization_divisor": INPUT_NORMALIZATION_DIVISOR,
            "all_zero_imagery_is_padding": True,
            "target_scaling": "none; source CHM height units",
            "output_nodata": -9999.0,
        },
        "training": {
            "batch_size": int(batch_size),
            "early_stopping_patience": int(early_stopping_patience),
            "initial_weights": os.path.abspath(pretrained_model) if pretrained_model else None,
            "initialization": "pretrained" if pretrained_model else "random",
            "optimizer_restored": False,
            "epoch_metrics_csv": os.path.abspath(os.path.splitext(weights_path)[0] + ".epoch_metrics.csv"),
            "metric_aggregation": "valid pixels; source CHM height units; training metrics during updates, validation in eval mode",
        },
        "checkpoint": {
            "epoch": int(epoch),
            "validation_loss": float(validation_loss),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    destination = os.path.splitext(weights_path)[0] + ".metadata.json"
    write_chm_json(destination, metadata)
    return destination


def write_chm_json(destination, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=os.path.dirname(os.path.abspath(destination)),
                                         prefix=".chm_report_", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            json.dump(payload, stream, indent=2, allow_nan=False)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary and os.path.isfile(temporary):
            os.remove(temporary)


def validate_pretrained_chm_path(pretrained_path, output_path):
    if not pretrained_path:
        return
    if not os.path.isfile(pretrained_path) or not pretrained_path.lower().endswith(".pth"):
        raise ValueError("Choose an existing pretrained .pth file.")
    if output_path:
        same = os.path.normcase(os.path.realpath(pretrained_path)) == os.path.normcase(os.path.realpath(output_path))
        if not same and os.path.exists(output_path):
            same = os.path.samefile(pretrained_path, output_path)
        if same:
            raise ValueError("Choose a different output .pth path so the pretrained weights are preserved.")


def load_pretrained_chm(model, pretrained_path):
    """Load complete compatible U-Net weights; do not silently skip layers."""
    metadata_path = os.path.splitext(pretrained_path)[0] + ".metadata.json"
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8-sig") as stream:
            metadata = json.load(stream)
        if (metadata.get("model_type") != "UNet" or metadata.get("input_channels") != 4
                or metadata.get("output_channels") != 1):
            raise ValueError("Pretrained metadata must describe a four-band, one-output CHM U-Net.")
        if metadata.get("preprocessing", {}).get("normalization_divisor") != INPUT_NORMALIZATION_DIVISOR:
            raise ValueError("Pretrained model metadata must use the fixed normalization divisor of 255.")
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Pretrained .pth must contain a PyTorch state dictionary.")
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, dict) or not state:
        raise ValueError("Pretrained checkpoint has no model state dictionary.")
    cleaned = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Pretrained state dictionary must map tensor names to tensors.")
        name = key.removeprefix("module.")
        if name in cleaned:
            raise ValueError(f"Duplicate pretrained tensor name: {name}")
        cleaned[name] = value
    expected = model.state_dict()
    if set(cleaned) != set(expected) or any(cleaned[name].shape != expected[name].shape for name in expected if name in cleaned):
        raise ValueError("Pretrained weights do not match this four-band CHM U-Net architecture. All tensor names and shapes must match.")
    model.load_state_dict(cleaned, strict=True)
    return len(cleaned)


def masked_height_error_sums(predictions, labels, valid_mask):
    """Return sums for pixel-weighted MAE, RMSE, and signed bias."""
    with torch.no_grad():
        errors = predictions.detach()[valid_mask].double() - labels[valid_mask].double()
        return errors.abs().sum().item(), errors.square().sum().item(), errors.sum().item()


def check_test_separation(training_dataset, test_dataset):
    """Reject files reused from training/validation, including hard links."""
    def keys(dataset):
        result = set()
        for folder in (dataset.image_dir, dataset.label_dir):
            for filename in dataset.filenames:
                path = os.path.join(folder, filename)
                result.add(("path", os.path.normcase(os.path.realpath(path))))
                stat = os.stat(path)
                if stat.st_ino:
                    result.add(("file", stat.st_dev, stat.st_ino))
        return result
    if keys(training_dataset).intersection(keys(test_dataset)):
        raise ValueError("Testing files must be separate from the imagery/CHM files used for training and validation.")


def evaluate_chm_test(model, dataset, batch_size, device):
    """Evaluate once; aggregate height errors over valid pixels, not batches."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    absolute_error = squared_error = signed_error = 0.0
    valid_pixels = evaluated_tiles = 0
    model.eval()
    arcpy.SetProgressor("step", "Evaluating optional test dataset...", 0, len(loader), 1)
    try:
        with torch.inference_mode():
            for index, (images, labels, valid_mask) in enumerate(loader, start=1):
                arcpy.SetProgressorLabel(f"Testing batch {index}/{len(loader)}")
                mask = valid_mask.to(device).bool()
                count = int(mask.sum().item())
                if count:
                    predictions = model(images.to(device))
                    labels = labels.to(device)
                    if predictions.shape != labels.shape:
                        raise ValueError("Test predictions and CHM labels have different dimensions.")
                    if not torch.isfinite(predictions[mask]).all():
                        raise ValueError("Nonfinite model predictions at valid testing pixels.")
                    errors = predictions[mask].double() - labels[mask].double()
                    absolute_error += errors.abs().sum().item()
                    squared_error += errors.square().sum().item()
                    signed_error += errors.sum().item()
                    valid_pixels += count
                    evaluated_tiles += int(valid_mask.flatten(1).any(1).sum().item())
                arcpy.SetProgressorPosition(index)
    finally:
        arcpy.ResetProgressor()
    if not valid_pixels:
        raise ValueError("Testing dataset contains no valid CHM pixels; no test metrics were written.")
    return {
        "tile_pairs": len(dataset), "evaluated_tiles": evaluated_tiles,
        "empty_tiles_skipped": len(dataset) - evaluated_tiles,
        "valid_pixels": valid_pixels, "mae": absolute_error / valid_pixels,
        "rmse": math.sqrt(squared_error / valid_pixels),
        "bias_predicted_minus_observed": signed_error / valid_pixels,
        "height_units": "source CHM units", "aggregation": "all valid tile pixels; overlapping pixels counted per tile",
    }


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
        img_files = {f for f in os.listdir(image_dir) if f.lower().endswith('.tif')}
        lbl_files = {f for f in os.listdir(label_dir) if f.lower().endswith('.tif')}
        self.filenames = sorted(img_files.intersection(lbl_files))
        self.unpaired_images = sorted(img_files - lbl_files)
        self.unpaired_labels = sorted(lbl_files - img_files)

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
        image /= INPUT_NORMALIZATION_DIVISOR

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

        test_img_dir = arcpy.Parameter(
            displayName="Test Images Directory (Optional, 4-Band .tif)", name="test_img_dir",
            datatype="DEFolder", parameterType="Optional", direction="Input")
        test_lbl_dir = arcpy.Parameter(
            displayName="Test Labels Directory (Optional, 1-Band CHM .tif)", name="test_lbl_dir",
            datatype="DEFolder", parameterType="Optional", direction="Input")
        pretrained_model = arcpy.Parameter(
            displayName="Pretrained Model Weights (.pth, Optional)", name="pretrained_model",
            datatype="DEFile", parameterType="Optional", direction="Input")
        pretrained_model.filter.list = ["pth"]
        early_stopping_patience = arcpy.Parameter(
            displayName="Early Stopping Patience (Epochs; 0 Disables)", name="early_stopping_patience",
            datatype="GPLong", parameterType="Required", direction="Input")
        early_stopping_patience.value = 5
        early_stopping_patience.filter.type = "Range"
        early_stopping_patience.filter.list = [0, 2147483647]
        return [img_dir, lbl_dir, model_type, epochs, batch_size, out_pth, test_img_dir, test_lbl_dir, pretrained_model, early_stopping_patience]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        if bool(parameters[6].valueAsText) != bool(parameters[7].valueAsText):
            parameters[6 if not parameters[6].valueAsText else 7].setErrorMessage(
                "Provide both test imagery and test CHM directories, or leave both empty.")
        if len(parameters) > 8 and parameters[8].valueAsText:
            try:
                validate_pretrained_chm_path(parameters[8].valueAsText, parameters[5].valueAsText)
            except (ValueError, OSError) as exc:
                parameters[8].setErrorMessage(str(exc))
        if len(parameters) > 9 and parameters[9].value is not None and int(parameters[9].value) < 0:
            parameters[9].setErrorMessage("Early stopping patience must be zero or greater.")

    def execute(self, parameters, messages):
        try:
            arcpy.SetProgressor("default", "Loading CHM datasets and initializing model...")
            self._execute(parameters, messages)
        finally:
            arcpy.ResetProgressor()

    def _execute(self, parameters, messages):
        image_dir = parameters[0].valueAsText
        label_dir = parameters[1].valueAsText
        model_choice = parameters[2].valueAsText
        num_epochs = parameters[3].value
        batch_size = parameters[4].value
        patience = int(parameters[9].value) if len(parameters) > 9 and parameters[9].value is not None else 5
        if patience < 0:
            raise ValueError("Early stopping patience must be zero or greater.")
        save_path = parameters[5].valueAsText
        pretrained_model = parameters[8].valueAsText if len(parameters) > 8 else None
        validate_pretrained_chm_path(pretrained_model, save_path)
        test_image_dir = parameters[6].valueAsText if len(parameters) > 6 else None
        test_label_dir = parameters[7].valueAsText if len(parameters) > 7 else None
        if bool(test_image_dir) != bool(test_label_dir):
            raise ValueError("Provide both test imagery and test CHM directories, or leave both empty.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arcpy.AddMessage(f"Starting training on device: {device}")

        # Pair imagery and CHM files by filename without header/alignment checks.
        full_dataset = CHMDataset(image_dir=image_dir, label_dir=label_dir)
        if full_dataset.unpaired_images or full_dataset.unpaired_labels:
            arcpy.AddWarning(f"Unpaired TIFF files excluded: {len(full_dataset.unpaired_images)} imagery, "
                             f"{len(full_dataset.unpaired_labels)} CHM. Pair names must match exactly.")
        arcpy.AddMessage(f"Total valid tile pairs found: {len(full_dataset)}")

        if len(full_dataset) == 0:
            arcpy.AddError("No matching .tif files found between image and label directories.")
            return

        test_dataset = None
        if test_image_dir:
            arcpy.AddMessage("Loading optional testing image/CHM pairs...")
            test_dataset = CHMDataset(test_image_dir, test_label_dir)
            if not len(test_dataset):
                raise ValueError("No matching TIFF pairs found in the testing directories.")
            if test_dataset.unpaired_images or test_dataset.unpaired_labels:
                arcpy.AddWarning(f"Unpaired testing files excluded: {len(test_dataset.unpaired_images)} imagery, "
                                 f"{len(test_dataset.unpaired_labels)} CHM.")
            check_test_separation(full_dataset, test_dataset)
            arcpy.AddMessage(f"Optional testing dataset: {len(test_dataset)} tile pairs; evaluated only after training.")
            
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

        if pretrained_model:
            arcpy.AddMessage(f"Loading pretrained CHM U-Net weights: {pretrained_model}")
            matched = load_pretrained_chm(model, pretrained_model)
            arcpy.AddMessage(f"Loaded {matched} model tensors. Starting a new optimizer and epoch counter.")

        # Initialize Loss, Optimizer, and Scheduler
        criterion = CombinedCHMLoss(alpha=0.85, data_range=40.0).to(device) 
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        epochs_completed = 0
        early_stopped = False
        arcpy.AddMessage(f"Early stopping patience: {patience} epochs (0 disables).")

        epoch_metrics_path = os.path.splitext(save_path)[0] + ".epoch_metrics.csv"
        epoch_fields = ["EPOCH", "TRAIN_LOSS", "VALIDATION_LOSS", "TRAIN_MAE", "TRAIN_RMSE",
                        "TRAIN_BIAS", "VALIDATION_MAE", "VALIDATION_RMSE", "VALIDATION_BIAS",
                        "LEARNING_RATE", "NEXT_LEARNING_RATE", "EPOCH_SECONDS", "TOTAL_SECONDS",
                        "TRAIN_TILE_PAIRS", "VALIDATION_TILE_PAIRS", "TRAIN_VALID_PIXELS",
                        "VALIDATION_VALID_PIXELS", "IS_BEST_CHECKPOINT", "EPOCHS_WITHOUT_IMPROVEMENT", "EARLY_STOPPED"]
        with open(epoch_metrics_path, "w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerow(epoch_fields)
        arcpy.AddMessage(f"Epoch metrics CSV: {epoch_metrics_path}")
        training_started = time.perf_counter()
        total_batches = num_epochs * (len(train_loader) + len(val_loader))
        completed_batches = 0
        arcpy.SetProgressor("step", "Training and validating CHM model...", 0, total_batches, 1)

        def advance_batch_progress():
            nonlocal completed_batches
            completed_batches += 1
            arcpy.SetProgressorPosition(completed_batches)

        # Training Loop
        for epoch in range(num_epochs):
            epoch_started = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            model.train() 
            train_loss = 0.0
            train_valid_pixels = 0
            train_absolute = train_squared = train_signed = 0.0
            
            for batch_idx, (images, labels, valid_mask) in enumerate(train_loader):
                arcpy.SetProgressorLabel(
                    f"Epoch {epoch + 1}/{num_epochs}: training batch {batch_idx + 1}/{len(train_loader)}")
                images, labels = images.to(device), labels.to(device)
                valid_mask = valid_mask.to(device)
                valid_count = int(valid_mask.sum().item())
                if not valid_count:
                    advance_batch_progress()
                    continue
                
                optimizer.zero_grad()
                predictions = model(images)
                loss = criterion(predictions, labels, valid_mask)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * valid_count
                train_valid_pixels += valid_count
                absolute, squared, signed = masked_height_error_sums(predictions, labels, valid_mask)
                train_absolute += absolute
                train_squared += squared
                train_signed += signed
                advance_batch_progress()
                
            if not train_valid_pixels:
                raise ValueError("Training partition contains no valid CHM pixels.")
            avg_train_loss = train_loss / train_valid_pixels
            
            model.eval() 
            val_loss = 0.0
            val_valid_pixels = 0
            val_absolute = val_squared = val_signed = 0.0
            
            with torch.no_grad(): 
                for batch_idx, (images, labels, valid_mask) in enumerate(val_loader):
                    arcpy.SetProgressorLabel(
                        f"Epoch {epoch + 1}/{num_epochs}: validation batch {batch_idx + 1}/{len(val_loader)}")
                    images, labels = images.to(device), labels.to(device)
                    valid_mask = valid_mask.to(device)
                    valid_count = int(valid_mask.sum().item())
                    if not valid_count:
                        advance_batch_progress()
                        continue
                    predictions = model(images)
                    loss = criterion(predictions, labels, valid_mask)
                    val_loss += loss.item() * valid_count
                    val_valid_pixels += valid_count
                    absolute, squared, signed = masked_height_error_sums(predictions, labels, valid_mask)
                    val_absolute += absolute
                    val_squared += squared
                    val_signed += signed
                    advance_batch_progress()
                    
            if not val_valid_pixels:
                raise ValueError("Validation partition contains no valid CHM pixels.")
            avg_val_loss = val_loss / val_valid_pixels
            scheduler.step(avg_val_loss)
            
            # Logging
            log_msg = f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            arcpy.AddMessage(log_msg)
            
            # Checkpointing
            is_best_checkpoint = avg_val_loss < best_val_loss
            if is_best_checkpoint:
                arcpy.SetProgressorLabel(f"Epoch {epoch + 1}/{num_epochs}: saving best checkpoint...")
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), save_path)
                metadata_path = save_chm_metadata(save_path, epoch + 1, best_val_loss, batch_size, pretrained_model, patience)
                arcpy.AddMessage(f"Model metadata saved to {metadata_path}")
                arcpy.AddMessage(f"  --> Best model saved to {save_path} (Val Loss improved to {best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
            early_stopped = patience > 0 and epochs_without_improvement >= patience

            arcpy.SetProgressorLabel(f"Epoch {epoch + 1}/{num_epochs}: writing epoch metrics...")
            # Close the file after every epoch so completed rows survive a later failure.
            now = time.perf_counter()
            with open(epoch_metrics_path, "a", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow([
                    epoch + 1, avg_train_loss, avg_val_loss,
                    train_absolute / train_valid_pixels, math.sqrt(train_squared / train_valid_pixels),
                    train_signed / train_valid_pixels, val_absolute / val_valid_pixels,
                    math.sqrt(val_squared / val_valid_pixels), val_signed / val_valid_pixels,
                    learning_rate, optimizer.param_groups[0]["lr"], now - epoch_started,
                    now - training_started, len(train_dataset), len(val_dataset),
                    train_valid_pixels, val_valid_pixels, is_best_checkpoint,
                    epochs_without_improvement, early_stopped,
                ])
            epochs_completed = epoch + 1
            if early_stopped:
                arcpy.AddMessage(f"Early stopping at epoch {epochs_completed}: validation loss has not improved "
                                 f"for {patience} epochs. Best validation loss: {best_val_loss:.4f}.")
                break

        if math.isfinite(best_val_loss):
            arcpy.SetProgressor("default", "Saving training summary...")
            metadata_path = os.path.splitext(save_path)[0] + ".metadata.json"
            with open(metadata_path, encoding="utf-8") as stream:
                metadata = json.load(stream)
            metadata["training"].update({
                "requested_epochs": int(num_epochs), "completed_epochs": epochs_completed,
                "early_stopped": early_stopped, "early_stopping_patience": patience,
                "stop_reason": "early_stopping" if early_stopped else "maximum_epochs",
            })
            write_chm_json(metadata_path, metadata)

        if test_dataset is not None:
            if not math.isfinite(best_val_loss):
                raise ValueError("No valid best checkpoint was saved; testing cannot proceed.")
            arcpy.SetProgressor("default", "Loading best checkpoint for testing...")
            arcpy.AddMessage("Loading the best validation checkpoint for testing...")
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            metrics = evaluate_chm_test(model, test_dataset, batch_size, device)
            arcpy.SetProgressor("default", "Saving test metrics and model metadata...")
            metadata_path = os.path.splitext(save_path)[0] + ".metadata.json"
            with open(metadata_path, encoding="utf-8") as stream:
                metadata = json.load(stream)
            report_path = os.path.splitext(save_path)[0] + ".test_metrics.json"
            metrics.update({"checkpoint_epoch": metadata["checkpoint"]["epoch"],
                            "model_weights": os.path.abspath(save_path),
                            "image_directory": os.path.abspath(test_image_dir),
                            "label_directory": os.path.abspath(test_label_dir),
                            "metrics_file": os.path.abspath(report_path)})
            write_chm_json(report_path, metrics)
            metadata["testing"] = metrics
            write_chm_json(metadata_path, metadata)
            arcpy.AddMessage(f"Testing (best checkpoint): MAE {metrics['mae']:.4f}, "
                             f"RMSE {metrics['rmse']:.4f}, bias {metrics['bias_predicted_minus_observed']:.4f} "
                             f"in source CHM units; {metrics['valid_pixels']} valid pixels.")
            arcpy.AddMessage(f"Testing metrics saved to {report_path}")
        arcpy.AddMessage("Training complete!")
        return

    def postExecute(self, parameters):
        return