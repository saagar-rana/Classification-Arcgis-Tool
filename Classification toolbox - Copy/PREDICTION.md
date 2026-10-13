# Predict Tree Species from Polygons

This tool classifies individual tree point clouds extracted from polygon footprints. It is available in `Point Cloud Classification.pyt` alongside the splitter and training tools.

## Inputs

1. **Input Tree Polygons**: add a tree polygon layer from the GeoPackage to ArcGIS Pro, then select that layer in the tool. Existing feature selections are respected. Use individual-tree polygons such as `Z1_polygons`, not the large `inference_zone` boundary polygons.
2. **Input LAZ/COPC File**: choose the matching acquisition date and zone, for example `2021-07-21-sbl-z1.copc.laz` for July 21 zone 1 polygons.
3. **Tree ID Field**: choose a unique, non-null identifier. The default is the source object ID. Its values are preserved in `DL_TREE_ID`; output object IDs may differ.
4. **Model Weights** and **Model Metadata**: use the corresponding `.pth` and `.metadata.json` from training. The tool suggests a metadata file beside the selected weights. Architecture, class order, sample size, and normalization come from metadata. Checkpoint loading is strict.
5. **Processing Area**, optional: restrict tree polygons to those intersecting this polygon layer. `inference_zone.gpkg` can be used here. It selects whole tree polygons; it does not replace their footprints or clip them to the area boundary.
6. **Reference Species Field**, optional: select `Label` to evaluate predictions against known species. Only successful predictions with reference labels matching model classes enter the metrics. Skipped polygons and unknown labels are counted separately.
7. **Batch Size**: defaults to 2; smaller batches reduce device memory use. Prediction supports a batch size of 1.
8. **Minimum Extracted Points**: defaults to 20. Trees below this threshold retain their polygons and receive a status rather than a prediction.
9. **Confidence Review Threshold**: defaults to 0.6. Predictions below this softmax score are marked for review. Scores are not calibrated probabilities.
10. **Sampling Seed**: defaults to 42. A deterministic per-tree seed produces consistent samples when data and settings are unchanged.
11. **Device**: Auto, CPU, or CUDA. CUDA requires an available compatible GPU.
12. **Output Polygon Feature Class**: choose a new feature-class name directly inside an existing file geodatabase.
13. **Report Folder**, optional: otherwise reports go beside the output geodatabase.

## Outputs

Original polygon attributes are retained. Added fields are:

| Field | Meaning |
| --- | --- |
| `DL_TREE_ID` | Preserved source identifier |
| `DL_SPECIES` | Predicted species |
| `DL_CONF` | Highest softmax score, 0–1 |
| `DL_NPOINTS` | Number of points inside the polygon before sampling |
| `DL_STATUS` | PREDICTED, NO_POINTS, TOO_FEW_POINTS, INVALID_GEOMETRY, or ERROR |
| `DL_REVIEW` | 1 for low-confidence predictions or skipped/error polygons; otherwise 0 |
| `DL_MODEL` | Model architecture |

For an output feature class called `Z1_species`, reports are:

```text
Z1_species_predictions.csv
Z1_species_prediction_summary.json
Z1_species.prediction_metrics.csv             # When reference evaluation is possible
Z1_species.prediction_confusion_matrix.csv    # When reference evaluation is possible
```

The predictions CSV also records error messages. Evaluation reports use the existing toolbox metric definitions. Reference accuracy is independent test performance only when the evaluated trees were held out from training and model selection.

## Extraction and preprocessing

The tool retrieves COPC candidates using each polygon's bounding box, then performs an exact XY polygon inclusion test, including boundaries and respecting holes. It retains all heights and LAS classification codes within the footprint; there is no ground removal, height threshold, or buffer in this version. Confirm this matches the extraction used for your training files.

XYZ, RGB, and intensity are required. Normalization follows version 1 of the training metadata. The saved number of points is sampled without replacement when possible, otherwise with replacement. Polygon and LAZ horizontal coordinate systems must match; project the polygons first if necessary.

Large clouds must use COPC. Plain LAS/LAZ is supported up to 5 million points and is loaded once. Very large COPC polygon bounding boxes are rejected using an estimated query-size guard; this is a safety estimate, not an exact point-count limit. Individual tree polygons should be used even when the processing area is large.

## Dependencies and checks

Requires the ArcGIS Pro environment, PyTorch, laspy with a LAZ backend such as lazrs, Shapely 2, and pyproj. Point Transformer V3 also needs its architecture-specific dependencies. Existing outputs require overwrite to be enabled. Source data is not edited; the polygon output is staged before completion.

Start with a small selection of tree polygons. Inspect point counts and statuses, compare polygon extraction with your training samples, and then process the full zone. Progress is reported by completed trees, including skipped polygons.

Core tests cover LAZ decompression, polygon boundaries and holes, normalization, strict model loading, inference, output orchestration, and reference reports. A real July 21 zone 1 polygon/COPC extraction was also checked. ArcGIS feature export and updates were tested with mocked ArcPy; a licensed ArcGIS Pro run is still required to verify those operations on your installation.
