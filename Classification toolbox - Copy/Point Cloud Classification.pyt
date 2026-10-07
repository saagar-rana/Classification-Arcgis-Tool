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
from datetime import datetime, timezone


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
        """Define the tool parameters."""
        # 1. Input Dataset Directory
        in_folder = arcpy.Parameter(
            displayName="Dataset Directory (Contains Class Folders)",
            name="in_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        selected_classes = arcpy.Parameter(
            displayName="Select Classes for Training",
            name="selected_classes",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        selected_classes.parameterDependencies = [in_folder.name]

        # 2. Number of Epochs
        epochs = arcpy.Parameter(
            displayName="Number of Epochs",
            name="epochs",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        epochs.value = 50 # Default value

        train_split = arcpy.Parameter(
            displayName="Training Data Split (0-1)",
            name="train_split",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        train_split.value = 0.8 # Default value

        model_names = ["PointNet", "PointTransformerV3", "DGCNN"]
        model_type = arcpy.Parameter(
            displayName="Model Type",
            name="model_type",
            datatype="GPString",
            parameterType="Required",   
            direction="Input"
        )
        model_type.filter.list = model_names
        model_type.value = "PointNet"

        batch_size = arcpy.Parameter(
            displayName="Batch Size",
            name="batch_size",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        batch_size.value = 20
        batch_size.filter.type = "Range"
        batch_size.filter.list = [2, 2147483647]

        patience = arcpy.Parameter(
            displayName="Patience for Early Stopping",
            name="patience",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        patience.value = 5 # Default value

        pretrained_model = arcpy.Parameter(
            displayName="Pretrained Model Weights (.pth)",
            name="pretrained_model",
            datatype="DEFile",  
            parameterType="Optional",
            direction="Input"
        )
        
        # 3. Output Weights File (.pth)
        out_pth = arcpy.Parameter(
            displayName="Output Model Weights (.pth)",
            name="out_pth",
            datatype="DEFile",
            parameterType="Required",
            direction="Output"
        )

        # Append the parameter to preserve the existing parameter indices.
        return [in_folder, selected_classes, epochs, train_split, model_type, patience, pretrained_model, out_pth, batch_size]

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        dir_path = parameters[0].valueAsText
        if not parameters[8].altered:
            parameters[8].value = 2 if parameters[4].valueAsText == "DGCNN" else 20
        classes = []
        if not dir_path or not os.path.isdir(dir_path):
            parameters[1].filter.list = classes
            return
        for folder_name in os.listdir(dir_path):
                    folder_path = os.path.join(dir_path, folder_name)
                    if os.path.isdir(folder_path):
                        for class_name in os.listdir(folder_path):
                            class_path = os.path.join(folder_path, class_name)
                            if os.path.isdir(class_path):
    
                                
                                if class_name not in classes:
                                    classes.append(class_name)

        parameters[1].filter.list = classes
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        if parameters[8].value is not None and int(parameters[8].value) < 2:
            parameters[8].setErrorMessage("Batch size must be at least 2 for training with batch normalization.")
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        dir_path = parameters[0].valueAsText
        selected_classes = parameters[1].valueAsText.split(";")
        num_epochs = parameters[2].value
        train_split = parameters[3].value
        model_type = parameters[4].valueAsText
        patience = parameters[5].value
        pretrained_model = parameters[6].valueAsText
        save_path = parameters[7].valueAsText
        batch_size = int(parameters[8].value)
        if batch_size < 2:
            raise ValueError("Batch size must be at least 2 for training with batch normalization.")

        # # Parse selected classes
        # selected_class_names = []
        # if selected_classes:
        #     # ArcGIS Pro passes multivalue strings as semicolon separated, sometimes with single quotes
        #     items = selected_classes.replace("'", "").split(";")
        #     # Extract just the class name before the " (X instances)"
        #     selected_class_names = [item.split(" (")[0].strip() for item in items]

        arcpy.AddMessage(f"Selected classes: {selected_classes}")
        file_paths = []
        classes = []
        total_classes = {}
        for folder_name in os.listdir(dir_path):
                    folder_path = os.path.join(dir_path, folder_name)
                    if os.path.isdir(folder_path):
                        for class_name in os.listdir(folder_path):
                            class_path = os.path.join(folder_path, class_name)
                            if os.path.isdir(class_path):
                                for file_name in os.listdir(class_path):
                                    print(f"Processing file: {file_name}")
                                    if file_name.endswith(".las"):
                                        file_paths.append(os.path.join(class_path, file_name))
                                
                                if class_name not in classes:
                                    classes.append(class_name)

        # Filter classes based on selected_classes
        if selected_classes:
            classes = [cls for cls in classes if cls in selected_classes]
            file_paths = [fp for fp in file_paths if fp.split('\\')[-2] in classes]
            arcpy.AddMessage(f"Filtered classes: {classes}")
            arcpy.AddMessage(f"Filtered file paths: {len(file_paths)} files")
            
        # print(f"Found {len(file_paths)} .las files across {len(classes)} classes: {classes}")


        for file in file_paths:
            name = file.split('\\')[-2]
            if name not in total_classes:
                total_classes[name] = 0
            total_classes[name] += 1


        arcpy.AddMessage(f"Found {len(file_paths)} .las files across {len(classes)} classes: {classes}")

        # Filter valid files (must have >= 4096 points)
        valid_paths = []
        for path in file_paths:
            try:
                las_file = laspy.read(path)
                if len(las_file.points) >= 4096:
                    valid_paths.append(path)
                del las_file
            except Exception as e:
                pass # Ignore problematic files

        final_paths = valid_paths
        arcpy.AddMessage(f"Retained {len(final_paths)} valid files with sufficient points.")

        if len(final_paths) == 0:
            arcpy.AddError("No valid .las files found with >= 4096 points.")
            return

        # --- Normalization & Data Loading ---
        def get_data(final_path, final_classes):
            num_points = 4096
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
            cls_name = os.path.basename(os.path.dirname(final_path))
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
             
            
        dataset = PointCloudDataset(final_paths, classes)
        
        train_size = int(train_split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
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
                "num_points": 4096,
                "minimum_source_points": 4096,
                "sampling": "random without replacement on each load",
                "coordinates": "subtract sampled mean x/y and sampled minimum z; no scale normalization",
                "rgb_divisor": 65535.0,
                "intensity": "subtract sampled median, divide by sampled IQR, then subtract minimum",
                "zero_iqr_replacement": 1e-6,
            },
            "training": {
                "requested_epochs": int(num_epochs),
                "batch_size": batch_size,
                "train_split": float(train_split),
                "early_stopping_patience": int(patience),
                "optimizer": "Adam",
                "initial_learning_rate": 0.001,
                "split_method": "random file split",
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

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (point_clouds, labels) in enumerate(train_loader):
                point_clouds, labels = point_clouds.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(point_clouds)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            arcpy.AddMessage(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

            model.eval()
            correct = 0
            total = 0
            test_loss = 0
            with torch.no_grad():
                for point_clouds, labels in test_loader:
                    point_clouds, labels = point_clouds.to(device), labels.to(device)
                    outputs = model(point_clouds)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_test_loss = test_loss / len(test_loader)
            accuracy = 100 * correct / total

            arcpy.AddMessage(f"Test Accuracy: {accuracy:.2f}%")
            arcpy.AddMessage(f"Test Loss: {avg_test_loss:.4f}")

            # Step the scheduler based on the validation loss
            scheduler.step(avg_test_loss)

            # Early stopping logic & model saving
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                metadata["checkpoint"] = {
                    "epoch": epoch + 1,
                    "validation_loss": float(avg_test_loss),
                    "validation_accuracy_percent": float(accuracy),
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                metadata_path = save_model_metadata(save_path, metadata)
                arcpy.AddMessage(f"Model weights saved to {save_path}")
                arcpy.AddMessage(f"Model metadata saved to {metadata_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    arcpy.AddMessage("Early stopping triggered.")
                    break

        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return

