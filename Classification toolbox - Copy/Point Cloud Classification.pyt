# -*- coding: utf-8 -*-

import arcpy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import laspy as las
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import laspy
# from ..pointnet.pointnet import PointNet
# from ..pointtransformerv3.pointtransformerv3 import PointTransformerV3

import sys
import json
import csv
import time
from datetime import datetime, timezone


def read_split_csv(filename):
    """Read explicit labels and memberships; reject duplicate or invalid rows."""
    splits = {"training": [], "validation": [], "testing": []}
    labels, seen = {}, set()
    with open(filename, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("Split CSV is empty.")
        reader.fieldnames = [name.strip().upper() for name in reader.fieldnames]
        if not {"FILE_PATH", "CLASS_NAME", "SPLIT"} <= set(reader.fieldnames):
            raise ValueError("CSV requires FILE_PATH, CLASS_NAME, and SPLIT columns.")
        for line, row in enumerate(reader, start=2):
            path = (row.get("FILE_PATH") or "").strip()
            species = (row.get("CLASS_NAME") or "").strip()
            subset = (row.get("SPLIT") or "").strip().lower()
            if not path or not species or subset not in splits:
                raise ValueError(f"Invalid path, class, or split on CSV line {line}.")
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(os.path.abspath(filename)), path)
            path = os.path.abspath(path)
            key = os.path.normcase(os.path.realpath(path))
            if key in seen:
                raise ValueError(f"Duplicate file across CSV rows: {path}")
            seen.add(key)
            splitter_module.validate_las(path, minimum_points=1)
            splits[subset].append(path)
            labels[path] = species
    if len(splits["training"]) < 2 or not splits["validation"]:
        raise ValueError("CSV needs at least two training files and one validation file.")
    classes = sorted({labels[path] for path in splits["training"]})
    unknown = {labels[path] for path in splits["validation"] + splits["testing"]} - set(classes)
    if unknown:
        raise ValueError(f"Validation/testing classes absent from training: {', '.join(sorted(unknown))}")
    return splits, labels, classes


def save_model_metadata(weights_path, metadata):
    """Write a JSON sidecar without changing the existing .pth format."""
    metadata_path = os.path.splitext(weights_path)[0] + ".metadata.json"
    temporary_path = metadata_path + ".tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, metadata_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return metadata_path


def write_evaluation_reports(weights_path, subset, classes, actual, predicted):
    """Export metrics in class-index order; zero-denominator metrics are zero."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Evaluation requires matching, nonempty actual/predicted labels.")
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for truth, guess in zip(actual, predicted):
        if not 0 <= truth < len(classes) or not 0 <= guess < len(classes):
            raise ValueError("Evaluation label is outside the saved class mapping.")
        matrix[truth, guess] += 1
    supports = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    metrics = []
    for index, name in enumerate(classes):
        tp = int(matrix[index, index])
        precision = tp / int(predicted_counts[index]) if predicted_counts[index] else 0.0
        recall = tp / int(supports[index]) if supports[index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.append((name, precision, recall, f1, int(supports[index]), int(predicted_counts[index])))
    total = len(actual)
    macro = [sum(row[i] for row in metrics) / len(classes) for i in (1, 2, 3)]
    weighted = [sum(row[i] * row[4] for row in metrics) / total for i in (1, 2, 3)]
    stem = os.path.splitext(weights_path)[0] + "." + subset
    metrics_path = stem + "_metrics.csv"
    confusion_path = stem + "_confusion_matrix.csv"
    with open(metrics_path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ROW_TYPE", "CLASS_NAME", "PRECISION", "RECALL", "F1", "SUPPORT", "PREDICTED_COUNT"])
        writer.writerows(("class", *row) for row in metrics)
        writer.writerow(["summary", "macro_average", *macro, total, total])
        writer.writerow(["summary", "weighted_average", *weighted, total, total])
    with open(confusion_path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ACTUAL_CLASS", *("PREDICTED:" + name for name in classes)])
        for name, counts in zip(classes, matrix):
            writer.writerow([name, *map(int, counts)])
    return {
        "samples": total,
        "accuracy_percent": 100 * int(np.trace(matrix)) / total,
        "macro_f1": macro[2],
        "weighted_f1": weighted[2],
        "metrics_csv": os.path.abspath(metrics_path),
        "confusion_matrix_csv": os.path.abspath(confusion_path),
        "metric_scale": "0 to 1; undefined precision/recall/F1 set to 0",
        "confusion_matrix_orientation": "rows actual, columns predicted",
    }

# 1. Get the exact path to the folder containing this .pyt file ("Classification toolbox")
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the sibling module by path so ArcGIS does not need it on sys.path.
import importlib.util
splitter_spec = importlib.util.spec_from_file_location(
    "quebec_tree_splitter", os.path.join(current_dir, "splitter.py"))
splitter_module = importlib.util.module_from_spec(splitter_spec)
splitter_spec.loader.exec_module(splitter_module)
SplitTreeDataset = splitter_module.SplitTreeDataset

# 2. Get the path to the parent folder ("DATASETS")
parent_dir = os.path.dirname(current_dir)

# 3. Add the DATASETS folder to Python's system path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 4. Now use ABSOLUTE imports (Notice there are no dots at the beginning!)
# Import architectures only when selected so optional dependencies do not
# prevent other models from running.

class Toolbox:
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Toolbox"
        self.alias = "toolbox"

        # List of tool classes associated with this toolbox
        self.tools = [SplitTreeDataset, Classification]


class Classification:
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Classify Trees"
        self.description = "Trains PointNet, Point Transformer V3, or DGCNN to classify individual tree point clouds."
        self.canRunInBackground = False

    def getParameterInfo(self):
        definitions = [
            ("Input Split CSV", "split_csv", "DEFile", "Required", "Input"),
            ("Number of Epochs", "epochs", "GPLong", "Required", "Input"),
            ("Model Type", "model_type", "GPString", "Required", "Input"),
            ("Batch Size", "batch_size", "GPLong", "Required", "Input"),
            ("Points per Tree", "num_points", "GPLong", "Required", "Input"),
            ("Patience for Early Stopping", "patience", "GPLong", "Required", "Input"),
            ("Pretrained Model Weights (.pth)", "pretrained_model", "DEFile", "Optional", "Input"),
            ("Output Model Weights (.pth)", "out_pth", "DEFile", "Required", "Output"),
        ]
        params = [arcpy.Parameter(displayName=label, name=name, datatype=dtype,
                                  parameterType=required, direction=direction)
                  for label, name, dtype, required, direction in definitions]
        params[0].filter.list = ["csv"]
        params[1].value = 50
        params[2].filter.list = ["PointNet", "PointTransformerV3", "DGCNN"]
        params[2].value = "PointNet"
        params[3].value = 20
        params[3].filter.type = "Range"
        params[3].filter.list = [2, 2147483647]
        params[4].value = 4096
        params[4].filter.type = "Range"
        params[4].filter.list = [20, 2147483647]
        params[5].value = 5
        params[6].filter.list = ["pth"]
        params[7].filter.list = ["pth"]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        if not parameters[3].altered:
            parameters[3].value = 2 if parameters[2].valueAsText == "DGCNN" else 20

    def updateMessages(self, parameters):
        for index, minimum, label in ((1, 1, "Epochs"), (3, 2, "Batch size"),
                                       (4, 20, "Points per tree"), (5, 1, "Patience")):
            if parameters[index].value is not None and int(parameters[index].value) < minimum:
                parameters[index].setErrorMessage(f"{label} must be at least {minimum}.")

    def execute(self, parameters, messages):
        try:
            self._execute(parameters, messages)
        finally:
            arcpy.ResetProgressor()

    def _execute(self, parameters, messages):
        split_csv = parameters[0].valueAsText
        num_epochs = int(parameters[1].value)
        model_type = parameters[2].valueAsText
        batch_size = int(parameters[3].value)
        num_points = int(parameters[4].value)
        patience = int(parameters[5].value)
        pretrained_model = parameters[6].valueAsText
        save_path = parameters[7].valueAsText
        if num_epochs < 1 or patience < 1 or batch_size < 2 or num_points < 20:
            raise ValueError("Use positive epochs/patience, batch size >=2, and points per tree >=20.")
        arcpy.SetProgressor("default", "Reading split CSV and checking LAS files...")
        splits, file_labels, classes = read_split_csv(split_csv)
        train_size = len(splits["training"])
        test_size = len(splits["validation"])
        total_classes = {name: sum(file_labels[path] == name for path in splits["training"]) for name in classes}
        arcpy.AddMessage(f"Classes from CSV: {classes}")
        arcpy.AddMessage(f"CSV splits: {train_size} training, {test_size} validation, {len(splits['testing'])} testing files. Memberships are preserved.")
        arcpy.AddMessage(f"Sampling {num_points} points per tree; smaller files are sampled with replacement.")

        # --- Normalization & Data Loading ---
        def get_data(final_path, final_classes):
            las_file = laspy.read(final_path)
            
            classification = np.copy(las_file.classification)
            x = np.copy(las_file.x)
            y = np.copy(las_file.y)
            z = np.copy(las_file.z)
            red = np.copy(las_file.red)
            green = np.copy(las_file.green)
            blue = np.copy(las_file.blue)
            intensity = np.copy(las_file.intensity)

            del las_file

            points = len(classification)
            replace_flag = False if points >= num_points else True
            indices = np.random.choice(points, num_points, replace=replace_flag)

            x = x[indices]
            y = y[indices]
            z = z[indices]
            red = red[indices]
            green = green[indices]
            blue = blue[indices]
            intensity = intensity[indices]

            # Normalize Coordinates
            n_coords = np.vstack((x, y, z)).T
            min_val = np.min(n_coords, axis=0)
            mean_val = np.mean(n_coords, axis=0)
            
            n_coords[:,0] -= mean_val[0]
            n_coords[:,1] -= mean_val[1]
            n_coords[:,2] -= min_val[2]

            # Normalize Colors
            colors = np.array([red, green, blue]).T / 65535.0

            # Normalize Intensity
            IQR = np.quantile(intensity, 0.75) - np.quantile(intensity, 0.25)
            if IQR == 0:
                IQR = 1e-6 
            n_intensity = (intensity - np.median(intensity)) / IQR
            n_intensity -= np.min(n_intensity)

            # Combine all 7 features into tensor
            data = np.hstack((n_coords, colors, n_intensity[:, np.newaxis]))
            cls_name = file_labels[final_path]
            class_idx = final_classes.index(cls_name)
        
            return data, class_idx

        
        class PointCloudDataset(Dataset):
            def __init__(self, final_paths, final_classes):
                self.final_paths = final_paths
                self.final_classes = final_classes

            def __len__(self):
                return len(self.final_paths)

            def __getitem__(self, idx):
                final_path = self.final_paths[idx]
                data, classification = get_data(final_path, self.final_classes)
                return torch.tensor(data, dtype=torch.float32), torch.tensor(classification, dtype=torch.long)  
             
            
        train_dataset = PointCloudDataset(splits["training"], classes)
        test_dataset = PointCloudDataset(splits["validation"], classes)
        
        if train_size < 2 or test_size < 1:
            raise ValueError("The data split must contain at least two training trees and one validation tree.")

        # Batch normalization in the classification heads requires >1 sample.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=(train_size % batch_size == 1))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # --- Model Configurations ---
        class PointNetConfig:
            def __init__(self):
                self.num_classes = len(classes)
                self.input_channels = 7     
                self.emb_dims = 1024        
                self.dropout = 0.3          
                self.feature_transform = False 

        pointnet_config = PointNetConfig()

        class PTv3Config:
            def __init__(self):
                self.num_classes = len(classes)
                self.channels_input = 7  
                self.grid_size = 0.01
                self.enc_depths = (2, 2, 2, 6, 2)
                self.enc_channels = (32, 64, 128, 256, 512)
                self.enc_num_head = (2, 4, 8, 16, 32)
                self.enc_patch_size = (48, 48, 48, 48, 48)
                self.stride = (2, 2, 2, 2)
                self.drop_path_rate = 0.3
                self.enable_flash = False 

        ptv3_config = PTv3Config()

        arcpy.SetProgressorLabel(f"Initializing {model_type}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arcpy.AddMessage(f"Using device: {device}")

        import importlib.util
        adapter_path = os.path.join(current_dir, "model_adapter.py")
        spec = importlib.util.spec_from_file_location("quebec_model_adapter", adapter_path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)

        if model_type == "PointNet":
            PointNet = adapter.load_local_model(parent_dir, "pointnet", "PointNet")
            model_config = vars(pointnet_config).copy()
            model = PointNet(config=pointnet_config).to(device)
        elif model_type == "PointTransformerV3":
            PointTransformerV3 = adapter.load_local_model(parent_dir, "pointtransformerv3", "PointTransformerV3")
            model_config = vars(ptv3_config).copy()
            model = PointTransformerV3(config=ptv3_config).to(device)
        elif model_type == "DGCNN":
            model = adapter.build_dgcnn(os.path.join(parent_dir, "models"), len(classes)).to(device)
            model_config = vars(model.config).copy()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        arcpy.AddMessage(f"Training {model_type} with batch size {batch_size}.")

        # Preserve the exact output-index order used by this training run.
        metadata = {
            "schema_version": 1,
            "model_type": model_type,
            "model_config": model_config,
            "weights_file": os.path.basename(save_path),
            "classes": list(classes),
            "class_to_index": {name: index for index, name in enumerate(classes)},
            "input_features": ["x", "y", "z", "red", "green", "blue", "intensity"],
            "preprocessing": {
                "version": 1,
                "num_points": num_points,
                "minimum_source_points": 1,
                "sampling": "random on each load; with replacement when source has fewer than num_points",
                "coordinates": "subtract sampled mean x/y and sampled minimum z; no scale normalization",
                "rgb_divisor": 65535.0,
                "intensity": "subtract sampled median, divide by sampled IQR, then subtract minimum",
                "zero_iqr_replacement": 1e-6,
            },
            "training": {
                "requested_epochs": int(num_epochs),
                "batch_size": batch_size,
                "train_split": train_size / (train_size + test_size),
                "early_stopping_patience": int(patience),
                "optimizer": "Adam",
                "initial_learning_rate": 0.001,
                "split_method": "CSV manifest",
                "split_csv": os.path.abspath(split_csv),
                "testing_samples": len(splits["testing"]),
                "train_samples": train_size,
                "validation_samples": test_size,
                "drop_last_training_batch": train_size % batch_size == 1,
            },
            "environment": {"torch_version": str(torch.__version__), "device": str(device)},
        }

        # Compute Class Weights for CrossEntropyLoss
        class_counts = [total_classes[c] for c in classes]
        total_samples = sum(class_counts)
        weights = [total_samples / (len(classes) * count) for count in class_counts]
        class_weights = torch.FloatTensor(weights).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        if pretrained_model and os.path.exists(pretrained_model):
            try:
                arcpy.AddMessage("Loading pre-trained weights...")
                pretrained_dict = torch.load(pretrained_model, map_location=device)
                
                # Check for nested state_dict often saved by PyTorch Checkpoints
                if 'state_dict' in pretrained_dict:
                    pretrained_dict = pretrained_dict['state_dict']
                    
                model_dict = model.state_dict()

                # Clean up parameter naming mismatch (e.g. from DDP 'module.')
                new_state_dict = {}
                for k, v in pretrained_dict.items():
                    new_key = k.replace('module.', '')
                    if new_key in model_dict and v.shape == model_dict[new_key].shape:
                        new_state_dict[new_key] = v

                model_dict.update(new_state_dict)
                model.load_state_dict(model_dict)
                arcpy.AddMessage(f"Pre-trained weights loaded successfully ({len(new_state_dict)} matched layers).")
            except Exception as e:
                arcpy.AddWarning(f"Failed to load pretrained model weights: {e}")

            

        # Initialize best loss and patience counter for early stopping
        best_loss = float('inf')
        patience_counter = 0
        epoch_metrics_path = os.path.splitext(save_path)[0] + ".epoch_metrics.csv"
        epoch_fields = ["EPOCH", "TRAIN_LOSS", "TRAIN_ACCURACY_PERCENT",
                        "VALIDATION_LOSS", "VALIDATION_ACCURACY_PERCENT",
                        "LEARNING_RATE", "NEXT_LEARNING_RATE", "EPOCH_SECONDS",
                        "TOTAL_SECONDS", "TRAIN_SAMPLES", "VALIDATION_SAMPLES",
                        "IS_BEST_CHECKPOINT", "EARLY_STOPPED"]
        with open(epoch_metrics_path, "w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerow(epoch_fields)
        metadata["training"]["epoch_metrics_csv"] = os.path.abspath(epoch_metrics_path)
        metadata["training"]["epoch_loss_aggregation"] = "mean of weighted cross-entropy batch losses"
        arcpy.AddMessage(f"Epoch metrics CSV: {epoch_metrics_path}")
        training_started = time.perf_counter()
        batches_per_epoch = len(train_loader) + len(test_loader)
        total_batches = num_epochs * batches_per_epoch
        completed_batches = 0
        arcpy.SetProgressor("step", "Training and validation...", 0, total_batches, 1)

        for epoch in range(num_epochs):
            epoch_started = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            model.train()
            running_loss = 0.0
            train_correct, train_total = 0, 0
            for i, (point_clouds, labels) in enumerate(train_loader):
                arcpy.SetProgressorLabel(
                    f"Epoch {epoch + 1}/{num_epochs}: training batch {i + 1}/{len(train_loader)}")
                point_clouds, labels = point_clouds.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(point_clouds)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                train_correct += (outputs.detach().argmax(dim=1) == labels).sum().item()
                train_total += labels.size(0)
                completed_batches += 1
                arcpy.SetProgressorPosition(completed_batches)
            
            arcpy.AddMessage(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

            model.eval()
            correct = 0
            total = 0
            test_loss = 0
            validation_actual, validation_predicted = [], []
            with torch.no_grad():
                for batch_index, (point_clouds, labels) in enumerate(test_loader, start=1):
                    arcpy.SetProgressorLabel(
                        f"Epoch {epoch + 1}/{num_epochs}: validation batch {batch_index}/{len(test_loader)}")
                    point_clouds, labels = point_clouds.to(device), labels.to(device)
                    outputs = model(point_clouds)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    validation_actual.extend(labels.cpu().tolist())
                    validation_predicted.extend(predicted.cpu().tolist())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    completed_batches += 1
                    arcpy.SetProgressorPosition(completed_batches)

            avg_test_loss = test_loss / len(test_loader)
            accuracy = 100 * correct / total

            arcpy.AddMessage(f"Validation Accuracy: {accuracy:.2f}%")
            arcpy.AddMessage(f"Validation Loss: {avg_test_loss:.4f}")

            # Step the scheduler based on the validation loss
            scheduler.step(avg_test_loss)

            # Early stopping logic & model saving
            is_best = avg_test_loss < best_loss
            if is_best:
                arcpy.SetProgressorLabel(f"Epoch {epoch + 1}/{num_epochs}: saving best checkpoint and validation reports...")
                best_loss = avg_test_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                metadata["validation"] = write_evaluation_reports(
                    save_path, "validation", classes, validation_actual, validation_predicted)
                metadata["checkpoint"] = {
                    "epoch": epoch + 1,
                    "validation_loss": float(avg_test_loss),
                    "validation_accuracy_percent": float(accuracy),
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                metadata_path = save_model_metadata(save_path, metadata)
                arcpy.AddMessage(f"Model weights saved to {save_path}")
                arcpy.AddMessage(f"Model metadata saved to {metadata_path}")
                arcpy.AddMessage(f"Best validation reports: {metadata['validation']['metrics_csv']} and {metadata['validation']['confusion_matrix_csv']}")
            else:
                patience_counter += 1
            early_stopped = patience_counter >= patience
            # Append and close each epoch so completed results survive interruptions.
            now = time.perf_counter()
            with open(epoch_metrics_path, "a", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow([
                    epoch + 1, running_loss / len(train_loader), 100 * train_correct / train_total,
                    avg_test_loss, accuracy, learning_rate, optimizer.param_groups[0]["lr"],
                    now - epoch_started, now - training_started, train_total, total,
                    is_best, early_stopped,
                ])
            if early_stopped:
                arcpy.AddMessage("Early stopping triggered.")
                break

        if splits["testing"]:
            if not np.isfinite(best_loss):
                raise ValueError("No finite best checkpoint was saved; testing cannot proceed.")
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            model.eval()
            testing_loader = DataLoader(PointCloudDataset(splits["testing"], classes), batch_size=batch_size, shuffle=False)
            arcpy.SetProgressor("step", "Evaluating held-out testing set...", 0, len(testing_loader), 1)
            testing_correct, testing_total = 0, 0
            testing_actual, testing_predicted = [], []
            with torch.no_grad():
                for batch_index, (point_clouds, labels) in enumerate(testing_loader, start=1):
                    arcpy.SetProgressorLabel(f"Testing batch {batch_index}/{len(testing_loader)}")
                    predictions = model(point_clouds.to(device)).argmax(dim=1)
                    testing_actual.extend(labels.tolist())
                    testing_predicted.extend(predictions.cpu().tolist())
                    testing_correct += (predictions == labels.to(device)).sum().item()
                    testing_total += len(labels)
                    arcpy.SetProgressorPosition(batch_index)
            testing_accuracy = 100 * testing_correct / testing_total
            arcpy.AddMessage(f"Held-out Testing Accuracy (best checkpoint): {testing_accuracy:.2f}%")
            arcpy.SetProgressorLabel("Saving testing reports and metadata...")
            metadata["testing"] = write_evaluation_reports(
                save_path, "testing", classes, testing_actual, testing_predicted)
            arcpy.AddMessage(f"Testing reports: {metadata['testing']['metrics_csv']} and {metadata['testing']['confusion_matrix_csv']}")
            save_model_metadata(save_path, metadata)
        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return

