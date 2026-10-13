"""Polygon-based tree species inference using saved toolbox checkpoints."""
import csv
import hashlib
import importlib.util
import json
import os
import tempfile
import uuid
from datetime import datetime
from collections import Counter
from types import SimpleNamespace

import arcpy
import laspy
import numpy as np
import torch

# The hosting toolbox supplies its existing evaluation-report writer.
report_writer = None
FEATURES = ["x", "y", "z", "red", "green", "blue", "intensity"]
RESULT_FIELDS = ["DL_SPECIES", "DL_CONF", "DL_NPOINTS", "DL_STATUS", "DL_REVIEW", "DL_MODEL"]


def horizontal_crs(crs):
    """Extract the native horizontal CRS from compound and bound definitions."""
    if crs.is_compound:
        components = [horizontal_crs(part) for part in crs.sub_crs_list]
        horizontal = [part for part in components if part.is_projected or part.is_geographic]
        if len(horizontal) != 1:
            raise ValueError("Coordinate system must contain exactly one horizontal CRS.")
        return horizontal[0]
    # A Bound CRS describes a transformation from its source to another datum.
    # Coordinates in the file still belong to the native source CRS.
    if crs.is_bound:
        return horizontal_crs(crs.source_crs)
    return crs.to_2d()


def load_model(weights, metadata_path, project_dir, device):
    with open(metadata_path, encoding="utf-8-sig") as stream:
        metadata = json.load(stream)
    classes = metadata.get("classes", [])
    if not classes or len(set(classes)) != len(classes) or not all(isinstance(x, str) and x for x in classes):
        raise ValueError("Metadata must contain a nonempty, unique ordered classes list.")
    if metadata.get("class_to_index") != {name: i for i, name in enumerate(classes)}:
        raise ValueError("Metadata class_to_index does not match its ordered classes.")
    preprocessing = metadata.get("preprocessing", {})
    if preprocessing.get("version") != 1 or metadata.get("input_features") != FEATURES:
        raise ValueError("This tool requires version 1 XYZ/RGB/intensity preprocessing metadata.")
    if int(preprocessing.get("num_points", 0)) < 1:
        raise ValueError("Metadata num_points must be positive.")
    config = metadata.get("model_config", {})
    if config.get("num_classes") != len(classes):
        raise ValueError("Model configuration and class mapping have different class counts.")
    kind = metadata.get("model_type")
    spec = importlib.util.spec_from_file_location(
        "quebec_prediction_adapter", os.path.join(os.path.dirname(__file__), "model_adapter.py"))
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    if kind == "PointNet":
        model_class = adapter.load_local_model(project_dir, "pointnet", "PointNet")
    elif kind == "PointTransformerV3":
        model_class = adapter.load_local_model(project_dir, "pointtransformerv3", "PointTransformerV3")
    elif kind == "DGCNN":
        model_class = adapter.load_dgcnn(os.path.join(project_dir, "models"))
    else:
        raise ValueError(f"Unsupported model architecture: {kind}")
    if config.get("input_channels", config.get("channels_input")) != 7:
        raise ValueError("Model must accept the seven saved input features.")
    model = model_class(config=SimpleNamespace(**config))
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, metadata


def prepare_points(points, metadata, seed):
    """Reproduce training normalization after sampling the polygon's points."""
    settings = metadata["preprocessing"]
    count = len(points)
    size = int(settings["num_points"])
    indices = np.random.default_rng(seed).choice(count, size, replace=count < size)
    coords = np.column_stack((np.asarray(points.x)[indices], np.asarray(points.y)[indices], np.asarray(points.z)[indices])).astype(np.float64)
    coords[:, :2] -= coords[:, :2].mean(axis=0)
    coords[:, 2] -= coords[:, 2].min()
    rgb = np.column_stack((np.asarray(points.red)[indices], np.asarray(points.green)[indices], np.asarray(points.blue)[indices])).astype(np.float64)
    divisor = float(settings.get("rgb_divisor", 65535.0))
    if divisor <= 0:
        raise ValueError("Metadata RGB divisor must be positive.")
    rgb /= divisor
    intensity = np.asarray(points.intensity)[indices].astype(np.float64)
    iqr = np.quantile(intensity, .75) - np.quantile(intensity, .25)
    if iqr == 0:
        iqr = float(settings.get("zero_iqr_replacement", 1e-6))
    intensity = (intensity - np.median(intensity)) / iqr
    intensity -= intensity.min()
    features = np.column_stack((coords, rgb, intensity)).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Preprocessed points contain nonfinite values.")
    return features


class PointSource:
    """Use indexed COPC queries; bound memory for unindexed LAS/LAZ files."""
    def __init__(self, filename):
        with laspy.open(filename) as reader:
            self.header = reader.header
        missing = {"X", "Y", "Z", "red", "green", "blue", "intensity"} - set(self.header.point_format.dimension_names)
        if missing:
            raise ValueError(f"Point cloud lacks required attributes: {', '.join(sorted(missing))}")
        self.is_copc = any(v.user_id == "copc" and v.record_id == 1 for v in self.header.vlrs)
        self.reader = None
        self.points = None
        if self.is_copc:
            self.reader = laspy.CopcReader.open(filename)
        else:
            if self.header.point_count > 5_000_000:
                raise ValueError("Unindexed LAS/LAZ exceeds 5 million points. Use the COPC LAZ file for this zone.")
            self.points = laspy.read(filename).points

    def extract(self, polygon):
        import shapely
        if not hasattr(shapely, "intersects_xy"):
            raise ValueError("Polygon prediction requires Shapely 2 or newer.")
        xmin, ymin, xmax, ymax = polygon.bounds
        if self.reader:
            cloud_area = (self.header.maxs[0] - self.header.mins[0]) * (self.header.maxs[1] - self.header.mins[1])
            if cloud_area > 0 and (xmax - xmin) * (ymax - ymin) / cloud_area * self.header.point_count > 10_000_000:
                raise ValueError("Polygon bounding box is too large for tree inference. Use individual-tree polygons rather than an inference-zone boundary.")
            bounds = laspy.copc.Bounds(np.array([xmin, ymin, self.header.mins[2]]),
                                       np.array([xmax, ymax, self.header.maxs[2]]))
            candidates = self.reader.query(bounds=bounds)
        else:
            x, y = np.asarray(self.points.x), np.asarray(self.points.y)
            candidates = self.points[(x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)]
        if not len(candidates):
            return candidates
        mask = shapely.intersects_xy(polygon, np.asarray(candidates.x), np.asarray(candidates.y))
        return candidates[mask]

    def close(self):
        if self.reader:
            self.reader.__exit__(None, None, None)


class PredictTreeSpecies:
    def __init__(self):
        self.label = "Predict Tree Species from Polygons"
        self.description = "Extract tree points from a LAZ/COPC cloud and predict species using saved weights and metadata."
        self.canRunInBackground = False

    def getParameterInfo(self):
        definitions = [
            ("Input Tree Polygons (GeoPackage Layer)", "tree_polygons", "GPFeatureLayer", "Required", "Input"),
            ("Input LAZ/COPC File", "point_cloud", "DEFile", "Required", "Input"),
            ("Tree ID Field", "tree_id_field", "Field", "Required", "Input"),
            ("Model Weights (.pth)", "model_weights", "DEFile", "Required", "Input"),
            ("Model Metadata (.json)", "model_metadata", "DEFile", "Required", "Input"),
            ("Processing Area (Optional)", "processing_area", "GPFeatureLayer", "Optional", "Input"),
            ("Reference Species Field (Optional)", "reference_field", "Field", "Optional", "Input"),
            ("Batch Size", "batch_size", "GPLong", "Required", "Input"),
            ("Minimum Extracted Points", "minimum_points", "GPLong", "Required", "Input"),
            ("Confidence Review Threshold (0-1)", "confidence_threshold", "GPDouble", "Required", "Input"),
            ("Sampling Seed", "sampling_seed", "GPLong", "Required", "Input"),
            ("Device", "device", "GPString", "Required", "Input"),
            ("Output Polygon Feature Class (File Geodatabase)", "out_features", "DEFeatureClass", "Required", "Output"),
            ("Report Folder (Optional)", "report_folder", "DEFolder", "Optional", "Input"),
            ("Output Predictions CSV", "predictions_csv", "DEFile", "Derived", "Output"),
        ]
        params = [arcpy.Parameter(displayName=label, name=name, datatype=dtype,
                                  parameterType=required, direction=direction)
                  for label, name, dtype, required, direction in definitions]
        params[0].filter.list = ["Polygon"]
        params[1].filter.list = ["laz", "las"]
        params[2].parameterDependencies = [params[0].name]
        params[3].filter.list = ["pth"]
        params[4].filter.list = ["json"]
        params[5].filter.list = ["Polygon"]
        params[6].parameterDependencies = [params[0].name]
        params[6].filter.list = ["Text"]
        params[7].value = 2
        params[8].value = 20
        params[9].value = .6
        params[10].value = 42
        params[11].filter.list = ["Auto", "CPU", "CUDA"]
        params[11].value = "Auto"
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        if parameters[3].valueAsText and not parameters[4].altered:
            metadata = os.path.splitext(parameters[3].valueAsText)[0] + ".metadata.json"
            if os.path.isfile(metadata):
                parameters[4].value = metadata
        if parameters[0].valueAsText and not parameters[2].altered:
            try:
                parameters[2].value = arcpy.Describe(parameters[0].valueAsText).OIDFieldName
            except Exception:
                pass

    def updateMessages(self, parameters):
        for index in (7, 8):
            if parameters[index].value is not None and int(parameters[index].value) < 1:
                parameters[index].setErrorMessage("Value must be at least 1.")
        if parameters[9].value is not None and not 0 <= float(parameters[9].value) <= 1:
            parameters[9].setErrorMessage("Confidence threshold must be between 0 and 1.")

    def execute(self, parameters, messages):
        source, layer, staging = None, None, None
        run_log = None

        def log_step(message):
            if run_log:
                # Close after each entry so a native process crash leaves evidence.
                with open(run_log, "a", encoding="utf-8") as stream:
                    stream.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")

        try:
            arcpy.SetProgressor("default", "Loading prediction model and checking inputs...")
            try:
                import shapely
                from shapely import wkb
                if not hasattr(shapely, "intersects_xy"):
                    raise ImportError("Shapely 2 or newer is required")
            except ImportError as exc:
                raise RuntimeError("Install Shapely 2 in the ArcGIS Pro environment to use polygon prediction.") from exc
            polygons = parameters[0].valueAsText
            cloud = parameters[1].valueAsText
            id_field = parameters[2].valueAsText
            weights = parameters[3].valueAsText
            metadata_path = parameters[4].valueAsText
            area = parameters[5].valueAsText
            reference = parameters[6].valueAsText
            batch_size = int(parameters[7].value)
            minimum = int(parameters[8].value)
            threshold = float(parameters[9].value)
            seed = int(parameters[10].value)
            requested_device = parameters[11].valueAsText
            if requested_device not in ("Auto", "CPU", "CUDA"):
                raise ValueError("Device must be Auto, CPU, or CUDA.")
            output = os.path.abspath(parameters[12].valueAsText)
            workspace, name = os.path.split(output)
            reports = parameters[13].valueAsText or os.path.dirname(workspace)
            if batch_size < 1 or minimum < 1 or not 0 <= threshold <= 1:
                raise ValueError("Use positive batch size/minimum points and confidence threshold between 0 and 1.")
            if not workspace.lower().endswith(".gdb") or not arcpy.Exists(workspace):
                raise ValueError("Choose an output feature class in an existing file geodatabase.")
            if not os.path.isdir(reports):
                raise ValueError("Report folder must exist.")
            stem = os.path.join(reports, name)
            report_paths = [stem + suffix for suffix in ("_predictions.csv", "_prediction_summary.json")]
            if reference:
                report_paths += [stem + ".prediction_metrics.csv", stem + ".prediction_confusion_matrix.csv"]
            if not arcpy.env.overwriteOutput and any(os.path.exists(path) for path in report_paths):
                raise ValueError("Prediction reports already exist. Choose a new output name/report folder or enable overwrite.")
            if arcpy.Exists(output) and not arcpy.env.overwriteOutput:
                raise ValueError("Output exists. Choose a new name or enable overwrite.")
            run_log = stem + "_prediction_run_" + uuid.uuid4().hex[:8] + ".log"
            log_step(f"START cloud={cloud}; weights={weights}; batch_size={batch_size}; requested_device={requested_device}")
            arcpy.AddMessage(f"Diagnostic run log: {run_log}")
            description = arcpy.Describe(polygons)
            if os.path.normcase(os.path.abspath(description.catalogPath)) == os.path.normcase(output):
                raise ValueError("Output must be different from the input polygon layer.")
            if description.shapeType != "Polygon":
                raise ValueError("Input tree layer must contain polygons.")
            existing = {field.name.upper() for field in arcpy.ListFields(polygons)}
            if existing.intersection(set(RESULT_FIELDS + ["DL_TREE_ID"])):
                raise ValueError("Input already contains reserved DL_ result fields. Select the original tree layer.")
            device = "cuda" if requested_device == "CUDA" or (requested_device == "Auto" and torch.cuda.is_available()) else "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA was selected but is unavailable.")
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_step(f"Loading model on {device}")
            model, metadata = load_model(weights, metadata_path, project_dir, device)
            log_step("Opening point cloud")
            source = PointSource(cloud)
            crs = source.header.parse_crs()
            if crs is None or not description.spatialReference.name or description.spatialReference.name == "Unknown":
                raise ValueError("Both point cloud and polygons must have known coordinate systems.")
            from pyproj import CRS
            exported_crs = description.spatialReference.exportToString()
            # ArcGIS appends grid-domain information after WKT. A semicolon can
            # also occur inside quoted WKT text, so do not split on semicolons.
            polygon_crs = CRS.from_wkt(exported_crs[:exported_crs.rfind("]") + 1])
            cloud_horizontal = horizontal_crs(crs)
            polygon_horizontal = horizontal_crs(polygon_crs)
            if not cloud_horizontal.equals(polygon_horizontal, ignore_axis_order=True):
                raise ValueError(
                    "Polygon and LAZ horizontal coordinate systems differ. "
                    f"Polygons: {polygon_horizontal.name}; LAZ: {cloud_horizontal.name}. "
                    "Project the polygons to the LAZ CRS first.")
            layer = "prediction_trees_" + uuid.uuid4().hex
            arcpy.management.MakeFeatureLayer(polygons, layer)
            if area:
                arcpy.management.SelectLayerByLocation(layer, "INTERSECT", area, selection_type="SUBSET_SELECTION")
            total = int(arcpy.management.GetCount(layer)[0])
            if not total:
                raise ValueError("No tree polygons are selected in the processing area.")
            identifiers = set()
            with arcpy.da.SearchCursor(layer, [id_field]) as cursor:
                for (identifier,) in cursor:
                    if identifier is None or str(identifier) in identifiers:
                        raise ValueError("Tree ID field must be non-null and unique in the selected polygons.")
                    identifiers.add(str(identifier))
            staging = os.path.join(workspace, "pred_work_" + uuid.uuid4().hex[:12])
            mappings = arcpy.FieldMappings()
            mappings.addTable(layer)
            identifier_map = arcpy.FieldMap()
            identifier_map.addInputField(layer, id_field)
            field = identifier_map.outputField
            field.name = "DL_TREE_ID"
            field.aliasName = "Source tree ID"
            identifier_map.outputField = field
            mappings.addFieldMap(identifier_map)
            arcpy.conversion.ExportFeatures(layer, staging, field_mapping=mappings)
            log_step(f"Exported {total} tree polygons; model={metadata['model_type']}; points_per_sample={metadata['preprocessing']['num_points']}")
            for field_name, dtype, length in [("DL_SPECIES", "TEXT", 255), ("DL_CONF", "DOUBLE", None),
                                              ("DL_NPOINTS", "DOUBLE", None), ("DL_STATUS", "TEXT", 32),
                                              ("DL_REVIEW", "SHORT", None), ("DL_MODEL", "TEXT", 128)]:
                arcpy.management.AddField(staging, field_name, dtype, **({"field_length": length} if length else {}))
            results, pending = {}, []
            classes = metadata["classes"]
            arcpy.AddMessage(f"Predicting {total} trees using {metadata['model_type']} on {device}; {metadata['preprocessing']['num_points']} points per sample.")
            arcpy.SetProgressor("step", "Extracting points and predicting trees...", 0, total, 1)
            completed = 0

            def flush():
                nonlocal completed
                if not pending:
                    return
                log_step(f"Inference START tree_ids={[item[2]['tree_id'] for item in pending]}; device={device}")
                batch = torch.from_numpy(np.stack([item[1] for item in pending])).to(device)
                try:
                    with torch.inference_mode():
                        probabilities = model(batch).softmax(dim=1).cpu().numpy()
                except torch.OutOfMemoryError as exc:
                    raise RuntimeError("Prediction ran out of device memory. Reduce Batch Size or select CPU.") from exc
                if probabilities.shape != (len(pending), len(classes)) or not np.isfinite(probabilities).all():
                    raise ValueError("Model produced invalid prediction probabilities.")
                for (oid, _, record), probabilities_row in zip(pending, probabilities):
                    index = int(probabilities_row.argmax())
                    record.update(species=classes[index], confidence=float(probabilities_row[index]),
                                  status="PREDICTED", review=int(probabilities_row[index] < threshold), predicted_index=index)
                    results[oid] = record
                    completed += 1
                pending.clear()
                log_step(f"Inference DONE completed={completed}/{total}")
                arcpy.SetProgressorPosition(completed)

            read_fields = ["OID@", "DL_TREE_ID", "SHAPE@"] + ([reference] if reference else [])
            with arcpy.da.SearchCursor(staging, read_fields) as cursor:
                for row in cursor:
                    oid, tree_id, geometry = row[:3]
                    record = dict(tree_id=str(tree_id), species=None, confidence=None, count=0,
                                  status="ERROR", review=1, reference=row[3] if reference else None, error="")
                    arcpy.SetProgressorLabel(f"Tree {tree_id}: extracting points ({completed}/{total} completed)")
                    try:
                        polygon = wkb.loads(bytes(geometry.WKB)) if geometry else None
                        if polygon is None or polygon.is_empty or not polygon.is_valid:
                            record["status"] = "INVALID_GEOMETRY"
                        else:
                            log_step(f"Extraction START tree_id={tree_id}; bounds={polygon.bounds}")
                            points = source.extract(polygon)
                            log_step(f"Extraction DONE tree_id={tree_id}; points={len(points)}")
                            record["count"] = len(points)
                            if len(points) < minimum:
                                record["status"] = "NO_POINTS" if not len(points) else "TOO_FEW_POINTS"
                            else:
                                stable_seed = int.from_bytes(hashlib.sha256(f"{seed}:{tree_id}".encode()).digest()[:8], "little")
                                pending.append((oid, prepare_points(points, metadata, stable_seed), record))
                    except Exception as exc:
                        record["error"] = str(exc)
                        arcpy.AddWarning(f"Tree {tree_id}: {exc}")
                    if not pending or pending[-1][0] != oid:
                        results[oid] = record
                        completed += 1
                        arcpy.SetProgressorPosition(completed)
                    if len(pending) >= batch_size:
                        flush()
            flush()
            arcpy.SetProgressor("default", "Writing polygon predictions and reports...")
            with arcpy.da.UpdateCursor(staging, ["OID@"] + RESULT_FIELDS) as cursor:
                for row in cursor:
                    result = results[row[0]]
                    cursor.updateRow([row[0], result["species"], result["confidence"], result["count"],
                                      result["status"], result["review"], metadata["model_type"]])
            stem = os.path.join(reports, name)
            csv_path = stem + "_predictions.csv"
            actual, predicted = [], []
            unknown_reference = 0
            class_indices = {species: i for i, species in enumerate(classes)}
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["TREE_ID", "PREDICTED_SPECIES", "CONFIDENCE", "POINT_COUNT", "STATUS", "REVIEW_FLAG", "REFERENCE_SPECIES", "ERROR"])
                for result in results.values():
                    writer.writerow([result["tree_id"], result["species"], result["confidence"], result["count"],
                                     result["status"], result["review"], result["reference"], result["error"]])
                    if reference and result["status"] == "PREDICTED":
                        if result["reference"] in class_indices:
                            actual.append(class_indices[result["reference"]])
                            predicted.append(result["predicted_index"])
                        else:
                            unknown_reference += 1
            summary = {"model_weights": os.path.abspath(weights), "model_metadata": os.path.abspath(metadata_path),
                       "point_cloud": os.path.abspath(cloud), "polygon_source": description.catalogPath,
                       "output_features": output, "counts": dict(Counter(r["status"] for r in results.values())),
                       "review_threshold": threshold, "sampling_seed": seed, "device": device,
                       "predictions_csv": csv_path, "unknown_or_blank_reference_labels": unknown_reference}
            if actual:
                if report_writer is None:
                    raise ValueError("Evaluation report writer is unavailable; open this tool through its toolbox.")
                summary["evaluation"] = report_writer(stem + ".pth", "prediction", classes, actual, predicted)
                arcpy.AddMessage(f"Reference-label accuracy: {summary['evaluation']['accuracy_percent']:.2f}% on {len(actual)} predicted trees.")
                arcpy.AddWarning("Reference evaluation may include training trees. Treat it as independent testing only if the input trees were held out.")
            elif reference:
                arcpy.AddWarning("No successful predictions have reference labels matching the model classes; no evaluation metrics were written.")
            with open(stem + "_prediction_summary.json", "w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2)
            if arcpy.Exists(output):
                arcpy.management.Delete(output)
            arcpy.management.Rename(staging, output)
            staging = None
            parameters[12].value = output
            parameters[14].value = csv_path
            arcpy.AddMessage(f"Output polygons: {output}")
            arcpy.AddMessage(f"Predictions CSV: {csv_path}")
            arcpy.AddMessage(f"Statuses: {summary['counts']}")
            log_step(f"SUCCESS statuses={summary['counts']}")
        except Exception as exc:
            log_step(f"FAILED {type(exc).__name__}: {exc}")
            raise
        finally:
            if source:
                source.close()
            for item in (staging, layer):
                if item and arcpy.Exists(item):
                    arcpy.management.Delete(item)
            arcpy.ResetProgressor()
