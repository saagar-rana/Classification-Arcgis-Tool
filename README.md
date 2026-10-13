# Forestry LiDAR Tools for ArcGIS Pro

Python toolboxes for preparing individual-tree LiDAR datasets, training tree-species classifiers, and predicting canopy height models (CHMs) from imagery.

The point cloud toolbox trains a classifier for one species label per tree file. Despite its current **Classify Trees** name, it is a training tool; it does not yet classify new tree files or assign species labels to every point in a forest cloud.

## Repository layout

```text
Classification toolbox - Copy/
    Point Cloud Classification.pyt
    model_adapter.py
    splitter.py               # Multiple-directory dataset splitting and CSV export
pointnet/                    # PointNet implementation used by the toolbox
pointtransformerv3/           # Point Transformer V3 implementation used by the toolbox
models/                      # Research architectures; DGCNN is integrated
Unet_CHM/
    Unet_Segmentor.pyt        # Train CHM Regressor
    Inference.pyt             # Predict CHM from Imagery
```

Keep these folders together. The classification toolbox loads model source files relative to its location. Keep `model_adapter.py` and `splitter.py` beside the `.pyt` file.

## Environment

Run the toolboxes in an ArcGIS Pro Python environment with `arcpy` available. Use a cloned Pro environment when adding dependencies.

The classification toolbox imports:

- PyTorch (`torch`)
- NumPy
- laspy
- pandas
- matplotlib

Point Transformer V3 also imports `torch_scatter`, `addict`, and `timm`, and uses `spconv` for sparse operations. Its configuration disables Flash Attention. Extension packages must be compatible with the Python, PyTorch, and CUDA versions in the environment. PointNet and DGCNN do not require these transformer extensions.

The CHM toolboxes additionally import `rasterio` and Hugging Face `transformers`.

CUDA is selected automatically when `torch.cuda.is_available()` is true; otherwise the classification tool uses the CPU. A CUDA-capable NVIDIA GPU is recommended for training. Check the packages from the Python environment selected in ArcGIS Pro:

```python
import arcpy
import torch
import laspy
import numpy
import pandas
import matplotlib

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

PointNet and DGCNN have passed small forward/backward checks in a local ArcGIS Pro Python environment with PyTorch 2.9.1. These checks do not establish compatibility with every Pro version or validate full dataset training. A pinned environment specification is not yet provided.

## Add the toolboxes to ArcGIS Pro

1. Download or clone the repository:

   ```powershell
   git clone https://github.com/saagar-rana/Forestry-lidar-tool.git
   ```

2. Open your ArcGIS Pro project.
3. In the Catalog pane, right-click **Toolboxes**, choose **Add Toolbox**, and select `Classification toolbox - Copy/Point Cloud Classification.pyt`.
4. Open **Split Tree Dataset** to prepare a split manifest, or **Classify Trees** to train from the folder layout described below.
5. Add either CHM `.pyt` file separately if you need the imagery workflow.

After editing a toolbox, refresh or reopen it. Restart ArcGIS Pro if it retains an older version of the code.

## Split Tree Dataset

This tool searches one or more dataset directories recursively for `.las` files. Each file's immediate parent folder supplies its class name. Overlapping input directories do not duplicate the same file in the manifest. LAS extensions are matched without regard to case in this tool.

| GUI parameter | Purpose | Default |
| --- | --- | --- |
| Dataset Directories | One or more folders to search recursively | Required |
| Select Classes | Classes discovered in the selected folders | Required; multiple selection |
| Training Fraction (0-1) | Requested fraction of files assigned to training in each class | 0.8 |
| Create Testing Set | Reserve an independent testing partition | False |
| Testing Fraction (0-1) | Requested testing fraction; enabled when testing is selected | 0.1 |
| Random Seed | Reproduce membership for the same valid file paths and settings | 42 |
| Output Split CSV | CSV filename in an existing output folder | Required |
| Minimum Points per File | Exclude files below this point count | 4096 |

Validation receives the remaining files. Training and testing fractions must leave a positive validation fraction. Each class needs at least two suitable files for training/validation, or three when testing is enabled. The complete manifest must contain at least two training files. Integer rounding and the reservation of at least one file per class in each enabled partition can make actual proportions differ from the requested fractions.

The tool checks the point count and required XYZ, RGB, and intensity dimensions in each LAS header. Unreadable or unsuitable files produce warnings and are skipped. Progress tracks files checked and CSV rows written. The output uses UTF-8 with a byte-order mark and contains:

```text
FILE_PATH,CLASS_NAME,SPLIT,DATASET_DIR,POINT_COUNT,RANDOM_SEED,TRAIN_FRAC,TEST_FRAC,MIN_POINTS
```

`FILE_PATH` is an absolute path. `SPLIT` is `training`, `validation`, or `testing`. `POINT_COUNT` records the original file point count; `MIN_POINTS` records the filtering threshold. The tool writes a temporary CSV before replacing the destination. Existing output requires ArcGIS overwrite to be enabled.

**Current integration limit:** Classify Trees does not read this CSV. It still takes a dataset directory and performs its own random split. The manifest's validation and testing membership therefore does not control the training tool. Keep the CSV for dataset tracking or use it with a separate workflow that explicitly honors its partitions.

Splitting is by file, separately within each class. Repeated observations of the same tree can enter different partitions; tree IDs and acquisition groups are not currently kept together.

## Prepare point cloud training data

The dataset must have two folder levels beneath its root: a grouping folder, such as an acquisition date, followed by species folders. Select the root as **Dataset Directory**.

```text
training_data/
    2021-05-28/
        ABBA/
            Tree_001.las
            Tree_002.las
        BEPA/
            Tree_003.las
    2021-06-17/
        ABBA/
            Tree_004.las
        BEPA/
            Tree_005.las
```

- Each file should contain an individual tree. The species folder supplies its label.
- Use at least two species for a meaningful classification experiment.
- For **Classify Trees**, files must use the lowercase `.las` extension. Neither dataset splitting nor training discovers `.laz` files.
- Each retained file must contain at least **4,096 points**.
- Files need XYZ coordinates, RGB attributes, and intensity. The current loader reads all seven features unconditionally.
- The split must leave at least two training files and one validation file. Use enough examples to represent every species in both subsets.

## Training parameters

| GUI parameter | Purpose | Default |
| --- | --- | --- |
| Dataset Directory (Contains Class Folders) | Root of the layout above | Required |
| Select Classes for Training | Species folders to include | Required |
| Number of Epochs | Maximum training epochs | 50 |
| Training Data Split (0-1) | Fraction of retained files used for training; remainder is validation | 0.8 |
| Model Type | PointNet, PointTransformerV3, or DGCNN | PointNet |
| Patience for Early Stopping | Consecutive epochs without improved validation loss before stopping | 5 |
| Pretrained Model Weights (.pth) | Optional weights; matching names and tensor shapes are loaded | None |
| Output Model Weights (.pth) | Destination for the best checkpoint | Required |
| Batch Size | Number of trees per training or validation batch | 20; automatically 2 for DGCNN until manually edited |

Batch size must be at least two because the classification heads use batch normalization during training. A final training batch containing only one tree is omitted. Reduce batch size if GPU memory is insufficient, particularly for DGCNN, which builds pairwise neighbor matrices.

### Run a first experiment

1. Select the dataset root and at least two species.
2. Choose **PointNet** for an initial check, or **DGCNN** to test the integrated research model.
3. Use a few epochs and a small batch size to check the workflow.
4. Choose an existing output folder and a filename such as `tree_model.pth`.
5. Run the tool and inspect the geoprocessing messages and saved files.

The tool uses Adam with an initial learning rate of `0.001`, weighted cross entropy, a learning-rate reduction scheduler, and early stopping based on validation loss. These settings are currently fixed except for the parameters listed above.

### Preprocessing

For each loaded tree, the training tool randomly samples 4,096 points without replacement and constructs features in this order:

```text
x, y, z, red, green, blue, intensity
```

It centers sampled X/Y coordinates, subtracts the sampled minimum Z, divides RGB by 65,535, and normalizes intensity using its sampled median and interquartile range before shifting the minimum to zero. Coordinates are not normalized to a unit scale. The metadata file records these rules.

## Outputs

Whenever validation loss improves, the tool saves:

```text
tree_model.pth
tree_model.metadata.json
```

The `.pth` is a PyTorch state dictionary. The JSON sidecar includes:

- Model type and architecture configuration.
- Ordered class names and the exact class-to-output-index mapping.
- Input features and preprocessing settings.
- Batch size, split fraction, sample counts, and other training settings.
- Saved checkpoint epoch, validation loss, validation accuracy, and UTC timestamp.
- PyTorch version and selected device.

Keep both files together. Class order follows discovery during that run; use the saved mapping rather than assuming alphabetical order. The metadata describes the best checkpoint, which may come from an earlier epoch than the final training epoch.

These outputs are custom PyTorch checkpoints, not Esri `.emd` or `.dlpk` packages. The repository does not currently include a separate point cloud prediction tool or export per-class evaluation reports.

## CHM tools

The repository also contains a separate imagery-to-CHM workflow:

- **Train CHM Regressor** in `Unet_CHM/Unet_Segmentor.pyt`: trains from directories of four-band imagery and matching one-band CHM labels.
- **Predict CHM from Imagery** in `Unet_CHM/Inference.pyt`: accepts four-band imagery, trained weights, a matching architecture, an input normalization divisor, and an output `.tif` path.

Available architecture choices are UNet, DPT (`Intel/dpt-large`), and DPT (`Intel/dpt-hybrid-midas`). This workflow predicts canopy height rather than delineating individual tree crowns. The point cloud metadata feature does not apply to these separate toolboxes.

## Current limitations

- The training/validation split is random by file. Repeated observations of the same tree may appear in both subsets. Grouped splitting and a separate held-out test set are not yet implemented.
- Split Tree Dataset can export a testing partition, but Classify Trees does not consume the manifest or evaluate that partition.
- The training sample size is fixed at 4,096 points. The splitter's configurable minimum controls file filtering only; it does not change the training sample size.
- Point sampling changes on every load, including validation; the tool does not currently expose a random seed.
- Messages labeled “Test Accuracy” and “Test Loss” refer to the validation subset used for model selection, not an independent test set.
- Read failures during the initial file check are silently skipped. RGB availability is not checked at that stage.
- Class weights use file counts collected before invalid files are removed and before splitting.
- Pretrained loading matches tensor names and shapes but does not yet validate species mappings or require a minimum number of matched layers.
- Other architectures in `models/` are research source files, not automatically available in the GUI. Their aggregate package imports depend on additional framework components and optional extensions.

## Troubleshooting

| Issue | What to check |
| --- | --- |
| `No module named 'pointnet'` | Use the updated toolbox and adapter; preserve the sibling `pointnet/` folder and reopen the tool. |
| Missing dependency | Install compatible packages in the environment ArcGIS Pro actually uses; a system Python installation is separate. |
| RGB attribute error | Confirm the LAS point format contains red, green, and blue attributes. |
| No valid files | Check folder nesting, lowercase `.las` extensions, readable files, and the 4,096-point minimum. |
| CUDA out of memory | Lower batch size; begin with two trees per batch for DGCNN. |
| One-class results | Include at least two species; a one-class model cannot distinguish alternatives. |
| Metadata not present | Confirm a best checkpoint was saved and that the updated toolbox is loaded. |
| Splitter class list is empty | Select existing dataset directories containing LAS files; the immediate parent folder of each file becomes its class. |
| Split fractions rejected | Keep training between zero and one and leave a positive validation remainder after testing. |
| CSV splits differ from training membership | The current training tool creates its own split and does not read the CSV manifest. |
| Prediction tool is missing | This checkout registers only Split Tree Dataset and Classify Trees in the point cloud toolbox. |

## Attribution and licensing

Model files include paper citations, upstream implementation references, and component license files. Preserve those notices when redistributing code. Check each component's license before reuse; this repository does not yet declare a single project-wide license.

Keep large LiDAR datasets, trained weights, local environments, and generated cache files outside source-control commits unless you have deliberately configured storage for them.
