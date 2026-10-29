# Forestry LiDAR Tools for ArcGIS Pro

Python toolboxes for preparing individual-tree LiDAR datasets, training tree-species classifiers, predicting species from tree polygons and point clouds, and estimating canopy height models (CHMs) from imagery.

## Available tools

| Toolbox | Tool | Purpose |
| --- | --- | --- |
| `Classification toolbox - Copy/Point Cloud Classification.pyt` | **Split Tree Dataset** | Search multiple dataset directories, select classes, and export training, validation, and optional testing memberships to CSV. |
| Same toolbox | **Classify Trees** | Train a tree-species classifier using the memberships and labels in the input CSV. |
| Same toolbox | **Predict Tree Species from Polygons** | Extract points inside individual-tree polygons from LAS/LAZ/COPC data and write species predictions to a file geodatabase. |
| `Unet_CHM/Unet_Segmentor.pyt` | **Train CHM Regressor** | Train an imagery-to-height model with epoch metrics, early stopping, and optional testing. |
| `Unet_CHM/Inference.pyt` | **Predict CHM from Imagery** | Predict a CHM raster and save it as a TIFF or file geodatabase raster. |

**Classify Trees is the training tool.** Use **Predict Tree Species from Polygons** to apply a trained classifier to new data. Classification assigns one species label per tree sample or polygon; it does not assign semantic labels to every point in a forest cloud. CHM regression predicts heights rather than delineating tree crowns.

## Repository layout

```text
Classification toolbox - Copy/
    Point Cloud Classification.pyt
    splitter.py                   # Multiple-directory splitting and CSV export
    model_adapter.py              # Loads the selected point cloud architecture
    pointnet2_torch.py            # Standard PyTorch PointNet++ SSG backend
    prediction.py                 # Polygon-based species prediction
    POINTNET2.md
    PREDICTION.md
pointnet/                         # PointNet backend
pointtransformerv3/               # Point Transformer V3 backend
models/                           # Research architectures; DGCNN is integrated
Unet_CHM/
    Unet_Segmentor.pyt            # CHM training
    Inference.pyt                 # CHM prediction
    chm_models.py                 # Shared model selection and checkpoint loading
    resunet.py                    # Residual U-Net
    unetplusplus.py               # U-Net++
    MODELS.md
```

Keep this folder structure when copying the tools. Keep the helper `.py` files beside their `.pyt` toolboxes. The point cloud models load relative to the repository; CHM helpers load from the toolbox directory.

## Environment and setup

Use an ArcGIS Pro Python environment with `arcpy` available. Clone the Pro environment before adding dependencies and select that environment in ArcGIS Pro.

| Workflow | Additional packages |
| --- | --- |
| Point cloud splitting and training | `torch`, `numpy`, `laspy`, `pandas`, `matplotlib` |
| Polygon species prediction | Above packages, a LAZ backend such as `lazrs`, Shapely 2, and `pyproj` |
| CHM training and prediction | `torch`, `numpy`, `rasterio` |
| Point Transformer V3 | Architecture-specific dependencies including `torch_scatter`, `addict`, `timm`, and `spconv` |

PointNet, DGCNN, and the integrated PointNet++ do not require the transformer extensions. PointNet++ uses standard PyTorch and does not require `pointnet2_ops`. The CHM models do not require Hugging Face `transformers`; DPT is no longer included.

Compiled extension packages must match the environment's Python, PyTorch, and CUDA versions. There is currently no pinned environment specification. Training and CHM prediction automatically choose CUDA when `torch.cuda.is_available()` is true, otherwise CPU. Polygon species prediction also exposes an explicit device selector.

Check the environment selected in ArcGIS Pro:

```python
import arcpy
import torch
import laspy
import rasterio

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

To add the tools:

1. Download the repository or clone it:

   ```powershell
   git clone https://github.com/saagar-rana/Forestry-lidar-tool.git
   ```

2. In ArcGIS Pro's Catalog pane, right-click **Toolboxes**, choose **Add Toolbox**, and select `Classification toolbox - Copy/Point Cloud Classification.pyt`.
3. Add `Unet_CHM/Unet_Segmentor.pyt` and `Unet_CHM/Inference.pyt` separately for the CHM workflow.
4. Refresh or reopen a toolbox after updates. Restart ArcGIS Pro if it retains an older version.

## Point cloud workflow

### 1. Prepare and split individual-tree files

Each source `.las` file should contain one tree and include XYZ, RGB, and intensity. The splitter searches recursively; the file's immediate parent folder supplies its class. Date or acquisition folders above the class folders are optional.

```text
tree_data/
    acquisition_1/
        ABBA/
            Tree_001.las
            Tree_002.las
        BEPA/
            Tree_003.las
    acquisition_2/
        ABBA/
            Tree_004.las
```

Open **Split Tree Dataset**:

| GUI parameter | Purpose | Default |
| --- | --- | --- |
| Dataset Directories | One or more folders to search recursively | Required |
| Select Classes | Discovered class names to include; supports multiple selection | Required |
| Training Fraction (0-1) | Requested training fraction within each class | 0.8 |
| Create Testing Set | Reserve a testing partition | False |
| Testing Fraction (0-1) | Requested testing fraction, enabled when testing is selected | 0.1 |
| Random Seed | Reproduce memberships for the same files and settings | 42 |
| Output Split CSV | CSV in an existing output folder | Required |
| Minimum Points per File | Exclude files below this point count | 4096 |

Validation receives the remainder after training and testing. Each class needs at least two suitable files, or three when testing is enabled. At least two training files are required overall. Rounding and the reservation of at least one file per class in each enabled partition can change the actual proportions.

The splitter checks LAS headers for point counts and required dimensions, warns about unsuitable files, and skips them. It deduplicates overlapping input directories. Progress tracks checked files and written rows. Existing output requires ArcGIS overwrite to be enabled.

The UTF-8 CSV includes:

```text
FILE_PATH,CLASS_NAME,SPLIT,DATASET_DIR,POINT_COUNT,RANDOM_SEED,TRAIN_FRAC,TEST_FRAC,MIN_POINTS
```

The classifier requires only `FILE_PATH`, `CLASS_NAME`, and `SPLIT`. Memberships must be `training`, `validation`, or `testing`. Splitter paths are absolute; a hand-prepared CSV can use paths relative to its own directory. For example:

```csv
FILE_PATH,CLASS_NAME,SPLIT
trees/ABBA/Tree_001.las,ABBA,training
trees/BEPA/Tree_002.las,BEPA,training
trees/ABBA/Tree_003.las,ABBA,validation
trees/BEPA/Tree_004.las,BEPA,validation
trees/ABBA/Tree_005.las,ABBA,testing
trees/BEPA/Tree_006.las,BEPA,testing
```

The CSV reader rejects duplicate file paths, invalid LAS files, and validation/testing classes that do not occur in training. Classes are sorted from training labels to define output indices. Use at least two species for a meaningful classification experiment.

**Splits are currently by file, not by tree ID or acquisition group.** Repeated observations of the same tree can cross partitions. Prepare a manifest that keeps related observations together when this matters. The training tool preserves the supplied CSV memberships.

### 2. Train a tree-species classifier

Open **Classify Trees** and select the split CSV. A separate dataset directory, class selector, and training split parameter are no longer used here.

| GUI parameter | Purpose | Default |
| --- | --- | --- |
| Input Split CSV | Labels and explicit training/validation/testing memberships | Required |
| Number of Epochs | Maximum training epochs | 50 |
| Model Type | `PointNet`, `PointTransformerV3`, `DGCNN`, or `PointNet++` | PointNet |
| Batch Size | Trees per batch | 20; suggested 2 for DGCNN or 4 for PointNet++ until manually changed |
| Points per Tree | Number of points sampled from each tree | 4096; minimum 20 |
| Patience for Early Stopping | Epochs without improved validation loss | 5; minimum 1 |
| Pretrained Model Weights (.pth) | Optional compatible weights | None |
| Output Model Weights (.pth) | Best-checkpoint destination | Required |

The batch size must be at least two because the classification heads use batch normalization. A final training batch with only one tree is omitted. Reduce batch size or Points per Tree if GPU memory is insufficient. DGCNN's pairwise neighbor calculations become more expensive as the point count increases.

The integrated PointNet++ uses single-scale grouping with radii 0.5 and 1.5 in source coordinate units. Its standard PyTorch farthest-point sampling favors compatibility over compiled-operator speed. See [PointNet++ details](Classification%20toolbox%20-%20Copy/POINTNET2.md).

Training uses Adam at an initial learning rate of `0.001`, inverse-frequency class weights calculated from training files, a validation-loss scheduler, and early stopping. Progress advances through training and validation batches. If testing rows exist, the best validation checkpoint is reloaded and evaluated on them after training.

#### Point cloud preprocessing

Each tree supplies seven features in this order:

```text
x, y, z, red, green, blue, intensity
```

The loader randomly samples the requested number of points without replacement when possible and with replacement for smaller files. The splitter's minimum-point threshold filters source files independently of Points per Tree. A custom CSV may reference smaller nonempty LAS files.

Preprocessing centers sampled X/Y coordinates, subtracts sampled minimum Z, divides RGB by 65,535, and normalizes intensity using its sampled median and interquartile range before shifting the minimum to zero. Coordinates retain their source scale. Point sampling changes on every load, including validation and testing; the split seed fixes membership, not sampled points.

#### Classification outputs

For output `tree_model.pth`:

```text
tree_model.pth
tree_model.metadata.json
tree_model.epoch_metrics.csv
tree_model.validation_metrics.csv
tree_model.validation_confusion_matrix.csv
tree_model.testing_metrics.csv             # When testing rows exist
tree_model.testing_confusion_matrix.csv    # When testing rows exist
```

Weights, metadata, and validation reports update when validation loss improves. The metadata saves the architecture configuration, ordered classes and class mapping, preprocessing, sample size, split CSV, training settings, and checkpoint details.

Per-class reports include precision, recall, F1, support, and predicted counts, with macro and weighted averages. Precision/recall/F1 use a 0-1 scale; accuracy uses percent. Undefined metrics are zero. Confusion matrices have actual classes in rows and predicted classes in columns.

The epoch CSV records training/validation loss and accuracy, learning rates before and after scheduling, elapsed time, evaluated sample counts, checkpoint status, and early stopping. Each completed epoch is appended and closed.

Keep the `.pth` and `.metadata.json` together. Classifier pretrained loading currently accepts matching tensor names and shapes and can load only part of a model; inspect the reported match count and ensure species mappings are appropriate. Prediction uses strict loading.

### 3. Predict species from tree polygons

Open **Predict Tree Species from Polygons**. Select individual-tree polygons from a GeoPackage layer and a matching point cloud, then supply the training `.pth` and `.metadata.json`. The metadata defines architecture, class order, point count, and normalization.

| Parameter | Default or requirement |
| --- | --- |
| Input Tree Polygons | Polygon layer; existing selection is respected |
| Input LAZ/COPC File | Matching acquisition and zone; LAS is also accepted |
| Tree ID Field | Unique, non-null identifier; source object ID is suggested |
| Model Weights / Model Metadata | Matching training outputs |
| Processing Area | Optional polygon layer selecting intersecting whole tree footprints |
| Reference Species Field | Optional labels for evaluation |
| Batch Size | 2; prediction also supports 1 |
| Minimum Extracted Points | 20 |
| Confidence Review Threshold | 0.6; softmax scores are not calibrated probabilities |
| Sampling Seed | 42; deterministic samples per tree |
| Device | Auto, CPU, or CUDA |
| Output Polygon Feature Class | New feature-class name directly inside an existing file geodatabase |
| Report Folder | Optional; otherwise reports go beside the output geodatabase |

Polygon and cloud horizontal coordinate systems must match. Project the polygons before prediction if needed. COPC allows spatial queries for large clouds; plain LAS/LAZ is limited to 5 million points and loaded once. Extraction retains all heights and LAS classification codes inside each polygon, including its boundary and excluding holes. Match this extraction to the way training trees were prepared.

The output retains polygon attributes and adds `DL_TREE_ID`, `DL_SPECIES`, `DL_CONF`, `DL_NPOINTS`, `DL_STATUS`, `DL_REVIEW`, and `DL_MODEL`. Skipped or failed trees retain a status. Reports include a predictions CSV and summary JSON; reference labels can additionally produce per-class metrics and a confusion matrix. Progress tracks completed trees.

See [polygon prediction inputs, outputs, and checks](Classification%20toolbox%20-%20Copy/PREDICTION.md) for the full workflow.

## Canopy height model workflow

### Supported CHM models

| Model Architecture | Parameters | Implementation |
| --- | ---: | --- |
| `UNet` | 31,038,209 | Existing U-Net, 64 starting channels |
| `ResUNet` | 8,114,561 | Residual encoder/decoder blocks, 32 starting channels; `resunet.py` |
| `UNet++` | 9,159,969 | Nested dense skip connections, 32 starting channels; `unetplusplus.py` |

These counts apply to the repository's four-input, one-output implementations. All produce nonnegative heights through a ReLU output. U-Net++ uses one final regression head without deep supervision. More detail is in [CHM model notes](Unet_CHM/MODELS.md).

### 1. Prepare paired imagery and labels

Supply four-band imagery and one-band CHM labels in separate folders with exactly matching `.tif` filenames:

```text
chm_data/
    images/
        tile_0001.tif
        tile_0002.tif
    labels/
        tile_0001.tif
        tile_0002.tif
```

Unpaired files are reported and excluded. Prepare imagery and labels with matching pixel grids, dimensions, coordinate systems, and band order; the tool does not run an image/CHM alignment validation. Use consistently sized training tiles. The original U-Net requires dimensions divisible by 16; the new models match decoder sizes to skip features for odd dimensions.

**Imagery is divided by 255 in both training and prediction.** The divisor is fixed and does not adapt to the source raster's bit depth. CHM labels retain their source height units; negative labels are clamped to zero.

Declared NoData, nonfinite values, and pixels where every imagery band is zero are excluded from valid training/evaluation pixels. Zero CHM height remains valid when imagery coverage is valid. An all-zero CHM label alone is not treated as missing data. Masked height loss and metrics exclude padding, including ordinary zero-filled borders from exported tiles.

### 2. Train a CHM regressor

Open **Train CHM Regressor**:

| GUI parameter | Purpose | Default |
| --- | --- | --- |
| Input Images Directory (4-Band .tif) | Training/development imagery | Required |
| Input Labels Directory (1-Band CHM .tif) | Matching CHM labels | Required |
| Model Architecture | `UNet`, `ResUNet`, or `UNet++` | UNet |
| Number of Epochs | Maximum epochs | 50 |
| Batch Size | Tile pairs per batch | 16 |
| Output Model Weights (.pth) | Best validation checkpoint | Required |
| Test Images Directory | Optional held-out test imagery | None |
| Test Labels Directory | Optional corresponding test labels; supply both test folders | None |
| Pretrained Model Weights (.pth, Optional) | Complete compatible weights for the selected architecture | None |
| Early Stopping Patience (Epochs; 0 Disables) | Consecutive epochs without lower validation loss | 5 |

Development pairs are split randomly into 80% training and 20% validation. The split is generated afresh for each run; the GUI does not expose a seed or fixed validation manifest. Optional test files must be separate from development files. They are evaluated once after the best validation checkpoint is restored. File separation does not establish geographic independence: use separate geographic zones when that is the evaluation goal.

Training uses a masked Smooth L1/SSIM loss, AdamW, and a validation-loss learning-rate scheduler. Progress reports training and validation batches, checkpoint saving, and optional testing. Pretrained loading checks metadata when present and requires all tensor names and shapes to match. It starts a new optimizer and epoch counter. Choose a different output filename to preserve the initial weights.

For output `chm_model.pth`:

```text
chm_model.pth
chm_model.metadata.json
chm_model.epoch_metrics.csv
chm_model.test_metrics.json    # When optional test folders are provided
```

Metadata records the architecture/configuration, fixed preprocessing, best checkpoint, training summary, and optional test results. The epoch CSV includes losses, MAE, RMSE, signed bias, learning rates, elapsed times, tile/valid-pixel counts, and early-stopping status. MAE, RMSE, and bias use source CHM height units and aggregate over valid pixels. Bias is predicted height minus observed height. Overlapping tiles count their shared pixels once per tile.

### 3. Predict a CHM raster

Open **Predict CHM from Imagery**, select the input imagery, matching weights, and the architecture used for training. The displayed normalization divisor is fixed at 255. Keep the metadata beside the weights so architecture and preprocessing mismatches can be detected.

Prediction reads the first four raster bands, processes overlapping 256 x 256 patches with cropped margins, and writes a one-band float32 height raster. Missing coverage receives NoData `-9999`. Output can be:

- A TIFF in an existing folder, such as `outputs/predicted_chm.tif`.
- A raster directly inside an existing file geodatabase, such as `outputs/results.gdb/predicted_chm`, **without a file extension**.

Existing output requires ArcGIS overwrite to be enabled. Keep `chm_models.py`, `resunet.py`, and `unetplusplus.py` beside both CHM toolboxes when distributing them.

## Limitations and validation

- The splitter does not automatically keep repeated trees or acquisition groups together. Prepare appropriate memberships before training.
- Classifier point samples vary during validation/testing; CHM development memberships vary between runs. Control these factors before interpreting small differences between model architectures.
- An evaluated test set remains useful only while it is excluded from training and model selection. Report species classification with accuracy and per-class metrics; report continuous CHM height errors with MAE, RMSE, and bias.
- PointNet++ neighborhood radii use the input coordinate units. Check their suitability for your cloud density and coordinate system.
- Other architectures in `models/` are research source files, not automatically available in the GUI. Some require additional frameworks or compiled extensions.
- Checkpoints are custom PyTorch weights with JSON sidecars, not Esri `.emd` or `.dlpk` packages.
- Model checks include small forward/backward and checkpoint tests. The CHM additions also passed short synthetic training, optional test evaluation, and TIFF prediction checks with mocked ArcPy. These checks do not establish full-dataset accuracy or verify every ArcGIS Pro version. Verify the GUI and licensed ArcPy operations on your installation, starting with a small dataset.

## Troubleshooting

| Issue | What to check |
| --- | --- |
| `No module named 'pointnet'` or missing local model file | Preserve the repository structure and sibling helpers, refresh the toolbox, and reopen it. |
| Missing dependency | Install compatible packages in the Python environment ArcGIS Pro actually uses. |
| No suitable LAS files | Splitter/training accept `.las`, case-insensitively. Check XYZ, RGB, intensity, and minimum points. LAZ is supported by the polygon predictor. |
| CSV rejected | Check required columns, split names, duplicate files, readable paths, training/validation counts, and classes missing from training. |
| CUDA out of memory | Lower batch size or classifier point count. DGCNN and dense U-Net++ features can consume substantial memory. |
| Splitter class list is empty | Select existing directories containing LAS files. Each file's immediate parent folder becomes its class. |
| Split fractions rejected | Leave a positive validation remainder and enough files per class for each enabled partition. |
| Polygon and LAZ coordinate systems differ | Project the polygons to the point cloud's horizontal CRS before prediction. |
| Most polygons have no or too few points | Check zone/date, coordinate systems, polygon placement, cloud coverage, and the minimum extracted point count. |
| CHM weights do not match | Select the training architecture and keep the matching metadata beside the weights. |
| CHM geodatabase output fails | Choose an existing `.gdb` and a valid raster name directly inside it, without `.tif`. |
| Unexpected CHM magnitudes | Confirm input imagery scaling matches fixed division by 255 and labels use the intended height units. |
| Older parameters or model list appear | Refresh/reopen the toolbox; restart ArcGIS Pro if necessary. |

## Attribution and licensing

Model files include paper citations, upstream implementation references, and component license files. Preserve those notices and check each component's license before reuse. The repository does not yet declare a single project-wide license.

Keep large LiDAR datasets, trained weights, local environments, and generated cache files outside source-control commits unless storage has been configured for them.
